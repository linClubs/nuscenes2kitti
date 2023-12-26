
import os

def create_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    image_path = os.path.join(save_dir, "images")
    cloud_path = os.path.join(save_dir, "points")
    calib_path = os.path.join(save_dir, "calibs")
    label_path = os.path.join(save_dir, "labels")
    pcd_path = os.path.join(save_dir, "pcd")

    if not os.path.exists(image_path):
        os.mkdir(image_path)

    if not os.path.exists(cloud_path):
        os.mkdir(cloud_path)

    if not os.path.exists(calib_path):
        os.mkdir(calib_path)

    if not os.path.exists(label_path):
        os.mkdir(label_path)
    
    if not os.path.exists(pcd_path):
        os.mkdir(pcd_path)


# save_dir = "/home/lin/ros_code/nus2bag_ws/src/nus_pkg/custom"
# create_save_dir(save_dir)