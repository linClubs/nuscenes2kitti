import numpy as np

# 外旋角度
theta_x = np.radians(30)
theta_y = np.radians(30)
theta_z = np.radians(30)

# 绕X轴外旋
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])

# 绕Y轴外旋
Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
               [0, 1, 0],
               [-np.sin(theta_y), 0, np.cos(theta_y)]])

# 绕Z轴外旋
Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
               [np.sin(theta_z), np.cos(theta_z), 0],
               [0, 0, 1]])

# 得到总的外旋矩阵
R_external = np.dot(Rz, np.dot(Ry, Rx))
print(R_external)
print("\n")
R_external = Rz @ Ry @ Rx
print(R_external)


# 内旋矩阵的构造与外旋相反

# 绕X轴内旋
Rx_internal = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), np.sin(theta_x)],
                        [0, -np.sin(theta_x), np.cos(theta_x)]])

# 绕Y轴内旋
Ry_internal = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)],
                        [0, 1, 0],
                        [np.sin(theta_y), 0, np.cos(theta_y)]])

# 绕Z轴内旋
Rz_internal = np.array([[np.cos(theta_z), np.sin(theta_z), 0],
                        [-np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]])

# 得到总的内旋矩阵

print("\n")
R_internal = np.dot(Rx_internal, np.dot(Ry_internal, Rz_internal))
print(R_internal)

print("\n")
R_internal =Rx_internal @ Ry_internal @ Rz_internal
print(R_internal)


from scipy.spatial.transform import Rotation as R
print("\n")
# 外旋
r_external = R.from_euler('zyx', [30, 30, 30], degrees=True)
print(r_external.as_matrix())
print(r_external.as_quat())

print("\n")
# 内旋
r_internal = R.from_euler('xyz', [30, 30, 30], degrees=True)
print(r_internal.as_quat())

