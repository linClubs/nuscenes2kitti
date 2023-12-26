#!/home/lin/software/miniconda3/envs/mmdet3d/bin/python
#coding=utf-8
import os
import numpy as np
# import open3d as o3d
import cv2
from nuscenes.nuscenes import NuScenes 
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from pypcd import pypcd

from create_dir import create_save_dir

# -----------参数-------------
# 数据集路径
dataroot = "/home/lin/code/datasets/nuscenes"
save_file = "/home/lin/ros_code/nus2bag_ws/src/nus_pkg/data/custom"
total_num = 404     # mini总共404帧

''' 
    categories10参数
    # 种类处理成10类,最开始统计为13类
    # bicycle_rack车架 合并到bicycle类  
    # pushable_pullable可推动的障碍合并到障碍barrier
    # debris残骸 去掉
'''
categories10 = True
#----------------------------

create_save_dir(save_file)

cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


if(categories10):
    nus_categories = ['car', 'truck', 'trailer', 'bus', 'construction',
                  'bicycle', 'motorcycle', 'pedestrian', 'trafficcone', 'barrier']
else:
# 按数量降序排列  bicycle_rack自行车架子  pushable_pullable可推动的障碍 debris残骸 垃圾堆
# trailer 拖车  barrier障碍物
    nus_categories = ['car', 'pedestrian', 'barrier', 'trafficcone', 'truck',
                    'motorcycle', 'bus', 'bicycle', 'construction', 'pushable_pullable', 
                    'trailer', 'bicycle_rack', 'debris']


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

# 文件名
count = 0

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

    '''
        每个样本都有n个标注, 每个sample的['anns']存储的n个token, 根据token就能确认标志信息
    '''
    print("anno_num:",  len(sample['anns']))
    
    # 将数据转到正常的lidar系 x向前  这里是幅度值
    q = Quaternion(axis=[0, 0, 1], angle = -1.57)  # 例子：绕 z 轴旋转 90 度
    T_new = np.eye(4)
    T_new[:3, :3] = q.rotation_matrix 
    T_lidar_world_new = T_new @ np.linalg.inv(T_global_lidar)

    # 点云转到新坐标
    points_new = points_hom @ T_new.T
    points_new = points_new[:, :3]
   
        # 2 相机数据处理   6个相机遍历
    imgs = []
    Ks = []
    lidar2cams = []
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
        
        T_cam_lidar = T_cam_global @ T_global_lidar

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
            
            # 投影点云 画图像上
            cv2.circle(img, (int(u), int(v)), 1, [0, 0, 255], -1, 16)
        
        imgs.append(img)
        Ks.append(K)
        lidar2cams.append(T_cam_lidar)

    '''
    3. 3dbox发布
        sample['anns']存储都是每帧lidar的n个标注物体的token
    '''
    # 记录kitti格式的标注
    label = ""

    for token in sample['anns']:
        annotation = nusc.get("sample_annotation", token)
        # print(annotation)
        T_ego_anno = get_T(annotation)
        anno_size = annotation['size']
        category_name = annotation['category_name'].split('.')[1]

        # bicycle_rack自行车架子  pushable_pullable可推动的障碍 debris残骸 垃圾堆
        if(categories10):
            if(category_name == 'bicycle_rack'):
                category_name = 'bicycle'
            if(category_name == 'pushable_pullable'):
                category_name = 'barrier'
            if(category_name == 'debris'):
                continue

        # nus封装好了Box类， 这些box都是global系下
        box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))
        
        # 3.1 box类 主要包含8个点（box）， center点和wlh, 和yaw
        # box的点坐标是3维 box.corners() 3 * 8
        box_conners = box.corners().T  # 转置后 8 * 3     
        box_center = box.center
        # wl表示dy和dx
        box_wlh = box.wlh
        box_q = box.orientation
        
        # 3.2 box的8个点变换齐次坐标，欧式变换到lidar系
        # box的点坐标变为齐次 8 * 4 
        box_conners = np.concatenate([box_conners, np.ones((len(box_conners), 1))], axis=1)
        
        # box 从global系变到 lidar系
        lidar_conners = box_conners @ np.linalg.inv(T_global_lidar).T
        # print(lidar_conners[:, :3])
        
        
        # 3.3 中心点变成4*1的维度, 并变换到雷达系
        box_center = np.append(box_center, 1).reshape([4, 1])
        # 中心点
        box_center_new =  T_lidar_world_new @ box_center
         # 中心点保留前3维度 xyz
        box_center_new = box_center_new[:3, :].reshape(-1)

        # 3.4 yaw的变换 
        # T_90° @ T_lidar_global @ yaw 这里nus的的lidar系x向右的，左乘T，将x轴变向前
        box_rotation_new = T_new[:3, :3] @ (np.linalg.inv(T_global_lidar))[:3, :3] @ box_q.rotation_matrix
        
        # 只取yaw角
        box_yaw_new = Quaternion(matrix=box_rotation_new).yaw_pitch_roll[0]
        
        # 3.5 box-dx，dy，dz，变换前后都不变，因为是box自身维参考系 
        dxyz= [box_wlh[1], box_wlh[0], box_wlh[2]]

        # 保留小数点位数
        # box_center_new = np.round(box_center_new, decimals=3)
        # dxyz = np.round(dxyz, decimals=3)
        # box_yaw_new = np.round(box_yaw_new, decimals=3)

        ss = np.concatenate((box_center_new, dxyz),axis=0)
        ss = np.append(ss, box_yaw_new)
        
        ss = np.round(ss, decimals=3)

        # 合成label
        for s in ss:
            label += str(s) + " " 
        
        label += category_name + "\n"

    # 4 保存数据
    ''' 
        坐标系：右手系 改117行左右的 q = Quaternion(axis=[0, 0, 1], angle = -1.57)
            angle为0是,使用的原版的nuslidar坐标系, angle= -1.57 统一的常规lidar系
        lidar: points_new Nx3维度 xyz 无i
        imgs, Ks, lidar2cams顺序: fl, f, fr, bl, b, br
        label: 是每帧的3d标签 x y z dx, dy, dz, yaw name
        count变量: 计数器, 用来定义不同的文件名
    '''
    
    # 4.1 点云增加强度i维度

    points_ = np.zeros([points_new.shape[0], 4], dtype=np.float32)
    points_[:, :3] = points_new   # 转换到常规lidar系下的点云数据
    points_[:, 3] = points[:, 3]  # 赋值强度信息
    
    flie_prefix = str(count).zfill(6)
    

    cloud_file_name = flie_prefix + ".bin"
    image_file_name = flie_prefix + ".png"
    label_file_name = flie_prefix + ".txt"
    calib_file_name = flie_prefix + ".txt"

    cloud_save_path = os.path.join(save_file, "points", cloud_file_name)
    label_save_path = os.path.join(save_file, "labels", label_file_name)
    calib_save_dir = os.path.join(save_file, "calibs")
    # 图像的父级目录
    image_save_parent_dir = os.path.join(save_file, "images")


    # 1 保存点云为bin格式
    with open(cloud_save_path, 'wb') as f:
        f.write(points_.tobytes())
    f.close()
    
    # 2 保存标签为 x y z dx, dy, dz, yaw name
    with open(label_save_path, 'w') as f:
        f.write(label)
    f.close()

    # 保存图像
    for i, im in enumerate(imgs):
        image_save_dir = os.path.join(image_save_parent_dir, "image" + str(i))
        if not os.path.exists(image_save_dir):
            os.mkdir(image_save_dir)
        
        image_save_path = os.path.join(image_save_dir, image_file_name)
        cv2.imwrite(image_save_path, im)


    # 4 保存标定参数
    calib_save_path = os.path.join(calib_save_dir, calib_file_name)
    # print(calib_save_path)   
    calib_texts = ""

    for i, j in enumerate(Ks):
        j = np.round(j[:3, :3].flatten(), decimals=4)
        calib_texts += "P" + str(i)
        
        for k, l in enumerate(j):
            calib_texts += ' ' + str(l)
        
        if i < 6:
            calib_texts += '\n'
       

    for i, lidar2cam in enumerate(lidar2cams):    
        lidar2cam = np.round(lidar2cam.flatten(), decimals=6)
        calib_texts += "lidarcam" + str(i)

        for j, k in enumerate(lidar2cam):
            calib_texts += ' ' + str(k)
        
        if i < 5:
            calib_texts += '\n'

    # print(calib_texts)
    with open(calib_save_path, 'w') as f:
        f.write(calib_texts)  
    f.close()
    
    # 5 点云保存为pcd格式
    # Create an example NumPy array with 'x', 'y', 'z' fields
    
    pcd_save_path = os.path.join(save_file, "pcd", flie_prefix + ".pcd")

    # print(len(points_))
    ##生成 pcd 的 head message
    '''
        FIELDS x y z intensity  # 定义点云中每个点包含的字段（属性）名称
        SIZE 4 4 4 4            # 每个字段的字节数，这里都是 4 字节（32 位浮点数）
        TYPE F F F F            # 每个字段的数据类型，这里都是浮点数（F 表示浮点数）
        COUNT 1 1 1 1           # 每个字段的元素数量，这里都是 1，表示每个字段是标量
        WIDTH 1000              # 点云数据的宽度，即点的数量
        HEIGHT 1                # 点云数据的高度，通常是 1，表示点云是一维的
        VIEWPOINT 0 0 0 1 0 0 0 # 视角信息
        POINTS 1000             # 点的总数量，与 WIDTH 相同
        DATA ascii / binary     # 数据的存储格式，这里是 ASCII
    '''

    # 生成 PCD 文件头信息
    meta_data = {
        'version': '0.7',
        'fields': ['x', 'y', 'z', 'intensity'],
        'size': [4, 4, 4, 4],
        'type': ['F', 'F', 'F', 'F'],
        'count': [1, 1, 1, 1],
        'width': points_.shape[0],  # 使用点云数据的行数作为宽度
        'height': 1,
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],
        'points': points_.shape[0],  # 使用点云数据的行数作为点数
        'data': 'ascii'
    }
   
    pcd_save = pypcd.PointCloud(meta_data, points_)
    pcd_save.save_pcd(pcd_save_path, compression='ascii')

    count += 1
    print("save nus2kitti number:  %d " %count)
    if count > total_num - 1:
        break

print(nus_categories)