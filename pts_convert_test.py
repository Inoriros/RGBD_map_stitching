import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import open3d as o3d
import RGBD2PointCloud as RGBD2PtsCloud
import h5py

# read point clouds
frame = 456
train_data = []
val_data = []
train_colors_data = []
val_colors_data = []
pts_num = 1

rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(int(frame))
pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
# save coordinates data
pts_dense = list(pcd_dense.points)
pts_sparse = list(pcd_sparse.points)
train_data.append(pts_dense)
val_data.append(pts_sparse)
# save RGB data
pts_colors_dense = list(pcd_dense.colors)
pts_colors_sparse = list(pcd_sparse.colors)
train_colors_data.append(pts_colors_dense)
val_colors_data.append(pts_colors_sparse)


train_data_ori = np.array(train_data) # data in wrong shape
val_data_ori = np.array(val_data) # val data in wrong shape
train_colors_data_ori = np.array(train_colors_data) # data in wrong shape
val_colors_data_ori = np.array(val_colors_data) # val data in wrong shape
len_train = train_data_ori.shape[1]
len_val = val_data_ori.shape[1]
train_data = train_data_ori.reshape((len_train, pts_num, 3))
val_data = val_data_ori.reshape((len_val, pts_num, 3))
train_colors_data = train_colors_data_ori.reshape((len_train, pts_num, 3))
val_colors_data = val_colors_data_ori.reshape((len_val, pts_num, 3))
print(train_data.shape)
print(val_data.shape)
train_XYZRGB_data = np.concatenate((train_data, train_colors_data), axis=2)
print(train_XYZRGB_data.shape)




# train_data_ori = np.array(train_data) # data in wrong shape
# val_data_ori = np.array(val_data) # val data in wrong shape
# train_colors_data_ori = np.array(train_colors_data) # data in wrong shape
# val_colors_data_ori = np.array(val_colors_data) # val data in wrong shape
# len_train = train_data_ori.shape[1]
# len_val = val_data_ori.shape[1]
# train_data = train_data_ori.reshape((len_train, pts_num, 3))
# val_data = val_data_ori.reshape((len_val, pts_num, 3))
# train_colors_data = train_colors_data_ori.reshape((len_train, pts_num, 3))
# val_colors_data = val_colors_data_ori.reshape((len_val, pts_num, 3))
# print(train_data.shape)
# print(val_data.shape)
# train_data = torch.from_numpy(train_data)
# val_data = torch.from_numpy(val_data)
# train_colors_data = torch.from_numpy(train_colors_data)
# val_colors_data = torch.from_numpy(val_colors_data)
# print(train_colors_data.shape)
# train_XYZRGB_data = torch.stack((train_data, train_colors_data), dim = -1)
# print(train_XYZRGB_data.shape)