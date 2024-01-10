import os
import shutil
import numpy as np
# import open3d as o3d
import cv2
from nuscenes.nuscenes import NuScenes 
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
# import sys
from pypcd import pypcd

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

# 创建路径
def make_dir(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

# ----------main从这里开始计算-----------------
# nus路径
dataroot = "/home/lin/code/mmdetection3d/data/nuscenes"
# 保存路径
saveroot = "/home/lin/ros_code/nus2bag_ws/src/nus_pkg/data/common"
save_flag = True   # 是否保存数据
categories10 = True  # 只取10类
# 相机名称
cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
# -----------------------------------------------

# 创建保存路径
make_dir(saveroot)

nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True) # 读取数据集

if(categories10):
    nus_categories = ['car', 'truck', 'trailer', 'bus', 'construction',
                  'bicycle', 'motorcycle', 'pedestrian', 'trafficcone', 'barrier']
else:
# 按数量降序排列  bicycle_rack自行车架子  pushable_pullable可推动的障碍 debris残骸 垃圾堆
# trailer 拖车  barrier障碍物
    nus_categories = ['car', 'pedestrian', 'barrier', 'trafficcone', 'truck',
                    'motorcycle', 'bus', 'bicycle', 'construction', 'pushable_pullable', 
                    'trailer', 'bicycle_rack', 'debris']

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
 
for ii, sample in enumerate(nusc.sample):
    # 1 雷达处理
    # 1.1 根据token取出lidar的信息
    lidar_token = sample['data']["LIDAR_TOP"]
    lidar_sample_data = nusc.get('sample_data', lidar_token)
    # 根据token取出雷达的文件名字
    lidar_file_name = lidar_sample_data['filename']
    lidar_path = os.path.join(dataroot, lidar_file_name)
    # print(lidar_file_name)
    
    ## 1.2 获取点云数据
    points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])
    # 强度信息
    points_intensity = points[:, 3]

    ## 1.3 齐次坐标
    '''
        点云转齐次坐标
        cloud原本是 n * 5(x,y,z,i,r) 取前3维度这里要去xyz并转成齐次坐标系
        np.ones(len(cloud), 1) 创建n行一列的1向量
    '''
    # 齐次坐标
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


    # 将数据转到正常的lidar系 x向前  这里是弧度制 1.57=90°
    # 原始的坐标是x向右，现在得把坐标系转90°。对应的点变换就是-90°
    q = Quaternion(axis=[0, 0, 1], angle = -1.57)  # 例子：把坐标系绕 z 轴旋转 -90 度
    T_new = np.eye(4)
    T_new[:3, :3] = q.rotation_matrix 
    # world -> nus_lidar -> lidar
    T_lidar_world_new = T_new @ np.linalg.inv(T_global_lidar)
    
    points_new = points_hom @ T_new.T   # 齐次坐标旋转90度

    # 2 相机数据处理   6个相机遍历
    img_front = np.ones([480, 640, 3])
    imgs_path = []  # 收集图像的路径，后续保存为kitti格式需要使用
    Ks = []         # 收集图像的内参，后续保存为kitti格式需要使用
    
    # lidar -> nus_lidar ->ego -> world -> ego-> cam   
    # 由于时间戳不用所以ego -> world不同，所有这里需要变换ego -> world -> ego
    Ts_cam_lidar_new = [] # 收集图像的外参，后续保存为kitti格式需要使用
    
    # 遍历cam
    for cam in cameras:
        # if(cam != "CAM_FRONT"):
        #     continue
        
        ## 2.1 获取相机token
        cam_token = sample['data'][cam]
        cam_sample_data = nusc.get("sample_data", cam_token)
        image_name = cam_sample_data['filename']

        ## 2.2 获取图像数据
        image_path = os.path.join(dataroot, image_name)
        imgs_path.append(image_path)
        
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
        Ks.append(K)
        # T_img_global
        T_img_global = K @ T_cam_global
        
        ## 2.4 获得像素系到雷达系的变换：T_img_lidar  lidar -> ego -> global -> ego -> cam -> img
        # 这个还是nus系
        T_img_lidar = T_img_global @ T_global_lidar
        
        # 这里就是将nus系转到正常的lidar系
        # lidar-> world -> cam  
        Ts_cam_lidar_new.append(T_cam_global @ np.linalg.inv(T_lidar_world_new))
                          
        # n * 4 @ 4 * 4 = n * 4 lidar点转到图像上
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
            
            # 点云投影到图像
            cv2.circle(img, (int(u), int(v)), 3, [0, 0, 255], -1, 16)
        
        # 保存前视的图像，后续用ros发布
        if(cam == "CAM_FRONT"):
            img_front = img
         
    cv2.imshow("front-view", img_front)
    if(cv2.waitKey(30) == ord('q')): 
        break
        # sys.exit()
    '''
        3 anno 标注信息解析 
        原始的nus标注是global系下
        可以将标注转到正常的雷达系和图像系  
        每个样本都有n个标注, 每个sample的['anns']存储的n个token, 根据token就能确认标志信息
        kitti标注格式: 
        正常标注格式: name x y z dx dy dz yaw
    '''
    print("anno_num:",  len(sample['anns']))

    # pass
    box_info = ''  # 定义一个需要写入txt文件的字符串
    for token in sample['anns']:
        annotation = nusc.get("sample_annotation", token)
        # print(annotation)
        T_ego_anno = get_T(annotation)  
        anno_size = annotation['size']
        category_name = annotation['category_name'].split('.')[1]

        # 过滤掉bicycle_rack,pushable_pullable,debris
        if(categories10):
            if(category_name == 'bicycle_rack'):
                category_name = 'bicycle'
            if(category_name == 'pushable_pullable'):
                category_name = 'barrier'
            if(category_name == 'debris'):
                continue

        # nus封装好了Box类， 这些box都是global系下
        box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))
        
        # box的中心点坐标是3维 box.corners() 3 * 8
        box_conners = box.corners().T  # 转置后 8 * 3

        box_center = box.center
        # wl表示dy和dx
        box_wlh = box.wlh
        box_q = box.orientation
        
        # box_yaw = (np.linalg.inv(T_global_lidar))[:3, :3] @ box.orientation.rotation_matrix
        # q = Quaternion(matrix=box_yaw).yaw_pitch_roll[0]

        # box的中心点坐标变为齐次 8 * 4 
        box_conners = np.concatenate([box_conners, np.ones((len(box_conners), 1))], axis=1)
        
        # box 从global系变到 nus的lidar系 
        lidar_conners = box_conners @ np.linalg.inv(T_global_lidar).T
        
        # box扩展一个维度变成4*1
        box_center = np.append(box_center, 1).reshape([4, 1])

        # 中心点，旋转yaw角, box-size 转到正常的lidar系下
        # 中心点 将world系下box转到正常的lidar系，相对于nus的lidar系 多转了90度
        box_center_new =  T_lidar_world_new @ box_center
        # 旋转yaw角 只去旋转部分是3*3格式
        box_rotation_new = T_new[:3, :3] @ (np.linalg.inv(T_global_lidar))[:3, :3] @ box_q.rotation_matrix
        box_yaw_new = Quaternion(matrix=box_rotation_new).yaw_pitch_roll[0]
        
        # yaw角限定到-π到π
        if box_yaw_new < -np.pi:
            box_yaw_new += (2*np.pi)
        if box_yaw_new > np.pi:
            box_yaw_new -= (2*np.pi)

        # print(box_center_new.shape, box.wlh.shape, box_yaw_new.shape)
        # 点取前3维度
        box_center_new = box_center_new[:3, :].reshape(-1)

        # box-size与坐标系无关，nus的封装的Box顺序是wlh， 注意kitti的格式顺序hwl，正常格式用lwh
        dxyz= [box_wlh[1], box_wlh[0], box_wlh[2]]

        # 拼接box内容 x y z dx dy dz yaw 
        ss = np.concatenate((box_center_new, dxyz), axis=0)
        ss = np.append(ss, box_yaw_new)

        # 标注信息更新 第一位为name， 遍历box内容生成字符串，并限制单位
        box_info += category_name
        for x in ss:
            box_info +=" " + str(np.round(x, 3))
        
        if(token != sample['anns'][-1]):  # txt标注文件中最后一行不需要\n
            box_info +="\n"

    # 5 可以通过nusc.get_sample_data获取boxes信息
    # 请注意，这些框被转换到当前传感器的坐标系中  这是是激光的token就box转到激光下
    # lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    # box_str = ""
    # for i, box in enumerate(boxes):
    #     label = box.label
    #     corners = box.corners().T
    #     locs = box.center   # 验证和上面world转到lidar系后的box一致
    #     wlh = box.wlh
    #     yaw = box.orientation.yaw_pitch_roll[0]
    #     yaw = np.array([yaw, ])   # yaw添加维度
    #     ss = np.concatenate((locs, wlh, yaw), axis=0)
    #     for s in ss:
    #         box_str += s.astype(str) + " " 
    #     print(box_str)
    #     break
    
    '''
    6 保存
    #  box数据：box_info 7维度得量 name x y z dx dy dz yaw
    # 点云数据：points_new[:, :3] 表示xyz, points_intensity点云的强度
    # 图片信息 imgs[6]  6个相机的图像数据  cameras标定相机顺序
    # Ks[6] 相机内参 4x4
    # Ts_cam_lidar_new  外参 T_cam_常规lidar  4x4
    '''
    # 如果保存数据 就创建文件夹
    if(save_flag):
        
        # 文件名字前缀 000000.xxx
        name_i = str(ii).zfill(6)

        # 1. 处理激光点云数据, n*4（x,y,z,i）
        lidar_dir_new = os.path.join(saveroot, "points")
        make_dir(lidar_dir_new)

        meta_data = {
            'version': '0.7',
            'fields': ['x', 'y', 'z', 'intensity'],
            'size': [4, 4, 4, 4],
            'type': ['F', 'F', 'F', 'F'],
            'count': [1, 1, 1, 1],
            'width': points_intensity.shape[0],  # 使用点云数据的行数作为宽度
            'height': 1,
            'viewpoint': [0, 0, 0, 1, 0, 0, 0],
            'points': points_intensity.shape[0],  # 使用点云数据的行数作为点数
            'data': 'ascii'
        }

        # (34720, 3) (34720,) 强度需要增加维度
        points_save = np.concatenate((points_new[:, :3], points_intensity[:,  np.newaxis]), axis=1)
        pcd_save = pypcd.PointCloud(meta_data, points_save)
        # 保存点云
        pcd_save.save_pcd(os.path.join(lidar_dir_new, name_i + ".bin"), compression='ascii')
        
        # 2. 保存box标注信息
        label_dir_new = os.path.join(saveroot, "labels")
        make_dir(label_dir_new)
        label_path_new = os.path.join(label_dir_new, name_i + ".txt")
        with open(label_path_new, 'w') as f:
            f.write(box_info)
        f.close()

        # 3. 保存传感器标定数据，内参和外参3*4， R0_rect为单位阵3*3
        calib_dir_new = os.path.join(saveroot, "calib")
        make_dir(calib_dir_new)
        calib_path_new = os.path.join(calib_dir_new, name_i + ".txt")
        
        calib_str = ''
        # 内参3x4
        for k in range(len(Ks)):
            K_str = [str(np.round(x, 3)) for x in Ks[k][:3, :4].reshape(-1)]
            calib_str += 'P' + str(k)+ ":" 
            for l in K_str:
               calib_str += (" "  + str(l))
            calib_str += "\n"
        # R0_rect:
        R0_rect = np.eye(3).reshape(-1)
        calib_str += "R0_rect:"
        for l in R0_rect:
               calib_str += (" "  + str(l))
        calib_str += "\n"

        # 外参3x4
        for k in range(len(Ts_cam_lidar_new)):
            T_str = [str(np.round(x, 3)) for x in Ts_cam_lidar_new[k][:3, :4].reshape(-1)]
            calib_str += 'T_lidar2cam' + str(k)+ ":" 
            for l in T_str:
               calib_str += (" "  + str(l))
            if(k < len(Ts_cam_lidar_new) -1):  # 最后一维不需要回车
                calib_str += "\n"
        
        with open(calib_path_new, 'w') as f:
            f.write(calib_str)
        f.close()

        # 4 保存图像
        # 创建图像的保存路径
        image_dir = os.path.join(saveroot, "images")
        make_dir(image_dir)

        for l, cam in enumerate(cameras):
            
            # 创建每个相机的目录
            cam_dir = os.path.join(image_dir, cam)
            make_dir(cam_dir)
            # 图像的目录
            img_path_new = os.path.join(cam_dir, name_i+'.png')
            # 不用opencv保存，速度太慢了，直接复制
            shutil.copy(imgs_path[l], img_path_new)
        #     cv2.imwrite(img_path_new, imgs[l])
        print("save %s frame" %(ii+1))        
