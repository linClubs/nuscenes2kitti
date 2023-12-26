#!/home/lin/software/miniconda3/envs/mmdet3d/bin/python
#coding=utf-8
import os
import numpy as np
import open3d as o3d
import cv2
from nuscenes.nuscenes import NuScenes 
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

import rospy

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from std_msgs.msg import Header


rospy.init_node('publish_point_cloud', anonymous=True)
rate = rospy.Rate(1)  # 发布频率1Hz
point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)

# q, t->T
def get_T(lidar_calibrator_data, inverse = False):
    T = np.eye(4, 4)
    T[:3, :3] = Quaternion(lidar_calibrator_data['rotation']).rotation_matrix
    T[:3, 3] = lidar_calibrator_data['translation']
    if inverse:
        T = np.linalg.inv(T)
    return  T


# 定义自定义COLORMAP，这里使用了一种简单的映射，从蓝色到红色
colormap = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)

# LIDAR_TOP
dataroot = "/home/lin/code/mmdetection3d/data/nuscenes"
cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True) # 读取数据集
samples = nusc.sample

# for sample in nusc.sample:
#     # 1 获取雷达数据
#     lidar_token = sample['data']["LIDAR_TOP"]    # 雷达token
#     # print(lidar_token)
#     lidar_sample_data = nusc.get('sample_data', lidar_token) # 雷达数据描述
#     lidar_file = os.path.join(dataroot, lidar_sample_data['filename'])  # 雷达路径


print(len(nusc.sample))  # 404个场景


# 统计点云的离激光的距离并计算max和min
lidar_dir = os.path.join(dataroot, "samples/LIDAR_TOP")
lidar_names = os.listdir(lidar_dir)

dist_arr = []
for lidar_name in lidar_names:
    path = os.path.join(lidar_dir, lidar_name)
    pc = np.fromfile(path, dtype=np.float32).reshape([-1, 5])[:, :3]
    dist = np.linalg.norm(pc, axis=1)  # 求2范数 也就是距离
    dist_arr.append(dist)
    # print(dist)

dist_arr = np.concatenate(dist_arr, dtype=np.float32)
# print(dist_arr)

max = np.max(np.array(dist_arr).reshape(-1))
min = np.min(np.array(dist_arr).reshape(-1))
print(max, min)

    
for sample in nusc.sample:
    # 1 雷达处理
    # 1.1 根据token取出lidar的信息
    lidar_token = sample['data']["LIDAR_TOP"]
    lidar_sample_data = nusc.get('sample_data', lidar_token)
    # 根据token取出雷达的文件名字
    lidar_file_name = lidar_sample_data['filename']
    lidar_path = os.path.join(dataroot, lidar_file_name)
    print(lidar_file_name)
    
    ## 1.2 获取点云数据
    points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])
    
    ## 1.3 齐次坐标
    '''
        点云转齐次坐标
        cloud原本是 n * 5(x,y,z,i,r) 取前3维度这里要去xyz并转成齐次坐标系
        np.ones(len(cloud), 1) 创建n行一列的1向量
    '''
    points_hom = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)

    # 1.4 获取T_ego_lidar, T_global_ego,  T_global_lidar
    # 获取lidar标定的token
    lidar_calibrator_data = nusc.get("calibrated_sensor", lidar_sample_data['calibrated_sensor_token'])
    
    # 雷达在ego系下pose Tego_lidar  lidar2ego
    T_ego_lidar = get_T(lidar_calibrator_data)
    ego_pose_data = nusc.get("ego_pose", lidar_sample_data['ego_pose_token']) 

    # ego在global系的坐标T_global_ego   ego2global
    T_global_ego = get_T(ego_pose_data)

    # T_global_lidar  lidar2global
    T_global_lidar = T_global_ego @ T_ego_lidar

    # 2 相机数据处理   6个相机遍历
    for cam in cameras:
        # if(cam != "CAM_FRONT"):
        #     continue
        
        ## 2.1 获取相机token
        cam_token = sample['data'][cam]
        cam_sample_data = nusc.get("sample_data", cam_token)
        image_name = cam_sample_data['filename']

        ## 2.2 获取图像数据
        image_path = os.path.join(dataroot, image_name)
        img = cv2.imread(image_path)

        ## 2.3 获取相机内参K, T_cam_ego, T_ego_global, T_img_global 
        # 相机时刻:T_global_ego ego2global  跟前面的 lidar时刻时ego2global要区分
        ego_pose_data = nusc.get("ego_pose", cam_sample_data['ego_pose_token'])
        
        # T_ego_global 这里给True，求逆  T_ego_global = T_global_ego.inv()
        T_ego_global2 = get_T(ego_pose_data, True)  #   global -> 当前相机ego         
        
        # 得到 T_ego_cam
        cam_calibrator = nusc.get("calibrated_sensor", cam_sample_data['calibrated_sensor_token'])
        
        # T_cam_ego = T_ego_cam.inv()
        T_cam_ego = get_T(cam_calibrator,  True)  #  ego-> cam
        
        # T_cam_global = T_cam_ego @ T_ego_global2
        T_cam_global = T_cam_ego @ T_ego_global2
    
        # 相机内参
        # 相机内参 3 * 3 
        K = np.eye(4, 4)
        K[:3, :3] = cam_calibrator['camera_intrinsic']

        # T_img_global
        T_img_global = K @ T_cam_global
        
        ## 2.4 获得像素系到雷达系的变换：T_img_lidar  lidar -> ego -> global -> ego -> cam -> img
        T_img_lidar = T_img_global @ T_global_lidar
        
        # n * 4 @ 4 * 4 = n * 4
        points_img = points_hom @ T_img_lidar.T
        # 此时还未除z，消除尺度   n * 4 @ 4 * 4 = n * 4
    
        # 相机归一化平面处理 / z ；前 2 维度xy分别除以z
        points_img[ :, :2] /= points_img[ : , [2]]

        # 排除z <= 0的像素点 第 3个维度得大于0
        for u, v, z in points_img[points_img[:, 2] > 0, :3]:
            # 排除图像范围外的点
            if u > img.shape[1] or u < 0 or v > img.shape[0] or v < 0: 
                continue

            p_img = np.array([u * z, u * v, z, 1], dtype=np.float32)
            p_lidar = p_img @ np.linalg.inv(T_img_lidar).T  # 4x1
            d = np.linalg.norm(p_lidar)  # 求2范数 也就是距离
            
            if(d >= max):
                d = max

            index = (d / max * 255).astype(np.uint8)
            
        
            cv2.circle(img, (int(u), int(v)), 1, [0, 0, 255], -1, 16)
        break
        # cv2.imshow("img", img)
        # cv2.waitKey()
        
    # 3 anno 标注信息解析 global系下  
    '''
        每个样本都有n个标注, 每个sample的['anns']存储的n个token, 根据token就能确认标志信息
    '''
    print("anno_num:",  len(sample['anns']))
    
    # 将数据转到正常的lidar系 x向前  这里是幅度值
    q = Quaternion(axis=[0, 0, 1], angle = 1.57)  # 例子：绕 z 轴旋转 45 度
    T_new = np.eye(4)
    T_new[:3, :3] = q.rotation_matrix 
    T_lidar_world_new = T_new @ np.linalg.inv(T_global_lidar)

    points_new = points_hom @ T_new.T
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"  # 坐标系

    # 定义点云数据结构
    pc_data = pc2.create_cloud_xyz32(header, points_new[:, :3])

    # 发布PointCloud2消息
    point_cloud_pub.publish(pc_data)
    rate.sleep()    

    pass
    for token in sample['anns']:
        annotation = nusc.get("sample_annotation", token)
        print(annotation)
        T_ego_anno = get_T(annotation)
        anno_size = annotation['size']
        category_name = annotation['category_name'].split('.')[1]
        # nus封装好了Box类， 这些box都是global系下
        box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))
        
        # box的点坐标是3维 box.corners() 3 * 8
        box_conners = box.corners().T  # 转置后 8 * 3

        box_center = box.center
        # wl表示dy和dx
        box_wlh = box.wlh
        box_q = box.orientation
        
        # box_yaw = (np.linalg.inv(T_global_lidar))[:3, :3] @ box.orientation.rotation_matrix
        # q = Quaternion(matrix=box_yaw).yaw_pitch_roll[0]

        # box的点坐标变为齐次 8 * 4 
        box_conners = np.concatenate([box_conners, np.ones((len(box_conners), 1))], axis=1)
        
        # box 从global系变到 lidar系
        lidar_conners = box_conners @ np.linalg.inv(T_global_lidar).T
        # print(lidar_conners[:, :3])
        
        

        box_center = np.append(box_center, 1).reshape([4, 1])
        # print(box_wlh)
        # box_wlh = box_wlh.reshape([3, 1])
        
        box_center_new =  T_lidar_world_new @ box_center
        # T_90° @ T_lidar_global @ yaw
        box_rotation_new = T_new[:3, :3] @ (np.linalg.inv(T_global_lidar))[:3, :3] @ box_q.rotation_matrix
        box_yaw_new = Quaternion(matrix=box_rotation_new).yaw_pitch_roll[0]
        
        print(box_center_new.shape, box.wlh.shape, box_yaw_new.shape)
        
        # 点取前3维度
        box_center_new = box_center_new[:3, :].reshape(-1)
        dxyz= [box_wlh[1], box_wlh[0], box_wlh[2]]
        ss = np.concatenate((box_center_new, dxyz),axis=0)
        ss = np.append(ss, box_yaw_new)
        break
    
    
    # 5 可以通过nusc.get_sample_data获取boxes信息
    # 请注意，这些框被转换到当前传感器的坐标系中  这是是激光的token就box转到激光下
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    
    box_str = ""
    for i, box in enumerate(boxes):
        label = box.label
        corners = box.corners().T
        locs = box.center   # 验证和上面world转到lidar系后的box一致
        wlh = box.wlh
        yaw = box.orientation.yaw_pitch_roll[0]
        yaw = np.array([yaw, ])   # yaw添加维度
        ss = np.concatenate((locs, wlh, yaw), axis=0)
        for s in ss:
            box_str += s.astype(str) + " " 
        print(box_str)
        break
    break