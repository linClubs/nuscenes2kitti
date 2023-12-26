import numpy as np
from scipy.spatial.transform import Rotation

def mat2euler(rotation):
    # 将旋转矩阵转换为欧拉角
    euler_zyx = rotation.as_euler('zyx', degrees=True)
    print("euler_zyx: ", euler_zyx)

    euler_xyz = rotation.as_euler('xyz', degrees=True)
    print("euler_xyz: ", euler_xyz)



r = Rotation.from_euler('xyz',[90, -90, 0], degrees=True)
rotation_matrix = r.as_matrix()


R_list = [[0, -1,  0],
     [0,  0, -1],
     [1,  0,  0]]

R = Rotation.from_matrix(R_list)
R_mat = R.as_matrix()
print(R_mat)


# 内旋模式 绕自身轴旋转
r1 = Rotation.from_euler('xyz',[90, -90, 0], degrees=True)
# rotation_matrix = r.as_matrix()
mat2euler(r1)

r2 = Rotation.from_euler('zyx',[0, -90, 90], degrees=True)
# rotation_matrix = r.as_matrix()
mat2euler(r2)


mat2euler(R)
