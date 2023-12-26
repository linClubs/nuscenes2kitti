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
from std_msgs.msg import Header
from cv_bridge import CvBridge

# --------------------
# 只发前视图跟雷达的消息
# dataroot = "/home/lin/code/mmdetection3d/data/nuscenes"
# --------------------

rospy.init_node('publish_cloud', anonymous=True)
dataroot = rospy.get_param('dataroot', default="/home/lin/ros_code/nus_ws/src/nus_pkg/data/nuscenes")

rate = rospy.Rate(10)  # 发布频率1Hz
pub_cloud = rospy.Publisher('/points_raw', PointCloud2, queue_size=10)
pub_img = rospy.Publisher('/image_raw', Image, queue_size=10)
cv_bridge = CvBridge()

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
    if(rospy.is_shutdown()):
        break
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

    '''
        每个样本都有n个标注, 每个sample的['anns']存储的n个token, 根据token就能确认标志信息
    '''
    print("anno_num:",  len(sample['anns']))
    
    # 将数据转到正常的lidar系 x向前  这里是幅度值
    q = Quaternion(axis=[0, 0, 1], angle = -1.57)  # 例子：绕 z 轴旋转 45 度
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
    pub_cloud.publish(pc_data)
    
    
        # 2 相机数据处理   6个相机遍历
    for cam in cameras:
        if(cam != "CAM_FRONT"):
            continue

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
        
        msg_img = cv_bridge.cv2_to_imgmsg(img, "passthrough")
        pub_img.publish(msg_img)


    rate.sleep() 

        # break
        # cv2.imshow("img", img)
        # cv2.waitKey()

