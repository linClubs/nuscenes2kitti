import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation


def make_dir(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

dataroot = "/home/lin/ros_code/nus2bag_ws/src/nus_pkg/data/common"
svaeroot = "/home/lin/ros_code/nus2bag_ws/src/nus_pkg/data/kitti"
cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

lidar_dir = os.path.join(dataroot, "points")
calib_dir = os.path.join(dataroot, "calib")
image_dir = os.path.join(dataroot, "images")
label_dir = os.path.join(dataroot, "labels")
lidar_names = os.listdir(lidar_dir)

make_dir(svaeroot)
lidar_dir_new = os.path.join(svaeroot, "points")
make_dir(lidar_dir_new)

calib_dir_new = os.path.join(svaeroot, "calib")
make_dir(calib_dir_new)

image_dir_new = os.path.join(svaeroot, "images")
make_dir(image_dir_new)

label_dir_new = os.path.join(svaeroot, "labels")
make_dir(label_dir_new)


for i, lidar_name in enumerate(lidar_names):
    name_prefix = lidar_name.split('.')[0]

    lidar_path = os.path.join(lidar_dir, lidar_name)
    lidar_path_new = os.path.join(lidar_dir_new, lidar_name)
    shutil.copy(lidar_path, lidar_path_new)

    calib_path = os.path.join(calib_dir, name_prefix + ".txt")
    calib_path_new = os.path.join(calib_dir_new, name_prefix + ".txt")
    shutil.copy(calib_path, calib_path_new)
    
    for cam in cameras:
        make_dir(os.path.join(image_dir_new, cam))
        image_path = os.path.join(image_dir, cam, name_prefix + ".png")
        # print(image_path)
        image_path_new = os.path.join(image_dir_new, cam, name_prefix + ".png")
        shutil.copy(image_path, image_path_new)
    
    label_path = os.path.join(label_dir, name_prefix + ".txt")
    label_path_new = os.path.join(label_dir_new, name_prefix + ".txt")
    

    infos = ""
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
        for line_ in lines:
            line = line_.strip().split(" ")
            # common的格式 name x y z dx dy dz 
            box = [float(x) for x in line[1:]]
            name = line[0]
            
            # print(line)
            kitti = {}
            kitti["objectType"] = name
            
            kitti["truncated"] = "1.0"
            kitti["occluded"] = "0"
            kitti["alpha"] = "0.0"

            # 2dbox
            kitti["bbox"] = [0.00, 0.00, 50.00, 50.00]  # should be higher than 50
            
            # common: lwh  ->  kitti: hwl
            kitti["diamensions"] = [box[5], box[4], box[3]] #height, width, length
            
            # xyz  kitti变成相机系 y向下
            xyz = np.array([box[0], box[1] , box[2]])
        
            # roll : 90
            # pitch: -90
            # yaw : 0
            
            # 0, -1,  0
            # 0,  0, -1
            # 1,  0,  0
            
            #  外, 这里内旋和外旋公式一致
            r = Rotation.from_euler('xyz',[90, -90, 0], degrees=True)
            rotation_matrix = r.as_matrix()
            # print(rotation_matrix)
            rotated_point = rotation_matrix @ xyz
            # print(rotated_point)

            # -90 0 -90 是坐标系变换雷达系变到相机系， 点变换就行T_lidar_cam, 
            # 如果是外旋就固定轴  如果是内旋就绕自身轴旋转    
            
            # np计算模式
            euler_angles = np.radians([90, -90, 0])

            R_x = np.array([[1, 0, 0],
                [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])

            R_y = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                            [0, 1, 0],
                            [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])

            R_z = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                            [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                            [0, 0, 1]])

            R = R_z @ R_y @ R_x
            rotated_point3 = R @ xyz
            # print(rotated_point3)

            # 可以直接手写变换 
            # kitti["location"] = [-box[1], -box[2] , box[0]] # camera coordinate
            # np.round保留小数点位数  y需要+z/2
            y = np.round(rotated_point[1] +box[5]/2, 3)
            kitti["locations"] = [np.round(rotated_point[0], 3), y, np.round(rotated_point[2], 3)]

            # 限定yaw角
            kitti["rotation_y"] = -box[-1]
            
            if kitti["rotation_y"] > 1.57:
                kitti["rotation_y"] -= 3.14
            if kitti["rotation_y"] < -1.57:
                kitti["rotation_y"] += 3.14
            
            # 限定小数点
            kitti["rotation_y"] = np.round(kitti["rotation_y"], 3)

            for key, value in kitti.items():
                if key == 'objectType':
                    infos += value
                elif (key == 'locations' or key == 'diamensions' or key == 'bbox'):
                    x_list = [str(x) for x in value]
                    for x in x_list:
                        infos += " " + x 
                else:
                    infos +=  " " + str(value)
            # 每行内容结束后增加一个回车
            if line_ != lines[-1]:
                infos +=  "\n"

    # f.close()            
    # print(infos)
    with open(label_path_new, 'w') as file:
        file.write(infos)
    file.close()
    
    print("save %s frame." %i)


     