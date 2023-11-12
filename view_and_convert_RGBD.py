import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import open3d as o3d
import RGBD2PointCloud as RGBD2PtsCloud
import h5py

# frame = 456
# rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
# pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
# # rgbd2PtsCloud.view_image()
# # rgbd2PtsCloud.view_PtsCloud()
# print(pcd_sparse)
# # Save pcd file
# # o3d.io.write_point_cloud('/Users/yidu/Desktop/UB_RA_Works/Super_Map/Data/P000/train/frame' + str(frame) + '.pcd', pcd_sparse, write_ascii=True, compressed=False, print_progress=False)

# pts_sparse_xyz = list(pcd_sparse.points)
# print(pts_sparse_xyz[0])



# region generate point cloud dataset

# # read point clouds
# train_data = []
# val_data = []
# frame_num = 300
# for frame in np.linspace(216, 216+frame_num-1, frame_num):
#     rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(int(frame))
#     pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
#     pts_dense = list(pcd_dense.points)
#     pts_sparse = list(pcd_sparse.points)
#     train_data.append(pts_dense)
#     val_data.append(pts_sparse)

# # Rearrange point cloud data size
# train_data_ori = np.array(train_data) # data in wrong shape
# len_train = train_data_ori.shape[1]
# train_data = train_data_ori.reshape((len_train, frame_num, 3))
# val_data_ori = np.array(val_data) # val data in wrong shape
# len_val = val_data_ori.shape[1]
# val_data = val_data_ori.reshape((len_val, frame_num, 3))
# # print(val_data_ori[0][0])
# # print(val_data[:][0][0])

# # save data into hdf5 file
# f = h5py.File("/Users/yidu/Downloads/shapenet.hdf5", "r+")
# print(list(f.keys()))
# hospital_dataset = f['02773838']
# # for 'train' data
# del f['02773838']['train'] # Delete the old 'train' point cloud data
# f['02773838'].create_dataset('train', data=train_data) # Add the new 'train' point cloud data
# # for 'val' data
# del f['02773838']['val'] # Delete the old 'val' point cloud data
# f['02773838'].create_dataset('val', data=val_data) # Add the new 'val' point cloud data
# print(list(hospital_dataset.keys()))
# # for 'test' data
# del f['02773838']['test'] # Delete the old 'test' point cloud data
# f['02773838'].create_dataset('test', data=val_data) # Add the new 'test' point cloud data
# print(list(hospital_dataset.keys()))

# endregion



# region generate RGB point cloud dataset
# read point clouds
train_data = []
val_data = []
train_colors_data = []
val_colors_data = []
frame_num = 300
for frame in np.linspace(216, 216+frame_num-1, frame_num):
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

# Rearrange point cloud data size
train_data_ori = np.array(train_data) # data in wrong shape
val_data_ori = np.array(val_data) # val data in wrong shape
train_colors_data_ori = np.array(train_colors_data) # data in wrong shape
val_colors_data_ori = np.array(val_colors_data) # val data in wrong shape
len_train = train_data_ori.shape[1]
len_val = val_data_ori.shape[1]
# Reshape
# train_data = train_data_ori.reshape((len_train, frame_num, 3))
# val_data = val_data_ori.reshape((len_val, frame_num, 3))
# train_colors_data = train_colors_data_ori.reshape((len_train, frame_num, 3))
# val_colors_data = val_colors_data_ori.reshape((len_val, frame_num, 3))
train_data = train_data_ori.reshape((frame_num, len_train, 3))
val_data = val_data_ori.reshape((frame_num, len_val, 3))
train_colors_data = train_colors_data_ori.reshape((frame_num, len_train, 3))
val_colors_data = val_colors_data_ori.reshape((frame_num, len_val, 3))
# print(train_data.shape)
# print(val_data.shape)
# Concatenate XYZ abd RGB together
train_XYZRGB_data = np.concatenate((train_data, train_colors_data), axis=2)
val_XYZRGB_data = np.concatenate((val_data, val_colors_data), axis=2)
print(train_XYZRGB_data.shape)
print(val_XYZRGB_data.shape)
# print(val_data_ori[0][0])
# print(val_data[:][0][0])



# SAVE data into hdf5 file
f = h5py.File("/home/jared/SAIR_Lab/Super-Map/diffusion-point-cloud-main/data/shapenet_flip.hdf5", "r+")
# 1. For XYZ pts only 
print(list(f.keys()))
hospital_dataset = f['02773838']
# (1) for 'train' data
del f['02773838']['train'] # Delete the old 'train' point cloud data
f['02773838'].create_dataset('train', data=train_data) # Add the new 'train' point cloud data
# (2) for 'val' data
del f['02773838']['val'] # Delete the old 'val' point cloud data
f['02773838'].create_dataset('val', data=val_data) # Add the new 'val' point cloud data
print(list(hospital_dataset.keys()))
# (3) for 'test' data
del f['02773838']['test'] # Delete the old 'test' point cloud data
f['02773838'].create_dataset('test', data=val_data) # Add the new 'test' point cloud data
print(list(hospital_dataset.keys()))

# 2. For XYZRGB pts 
print(list(f.keys()))
hospital_XYZRGB_dataset = f['02801938']
# (1) for 'train' data
del f['02801938']['train'] # Delete the old 'train' point cloud data
f['02801938'].create_dataset('train', data=train_XYZRGB_data) # Add the new 'train' point cloud data
# (2) for 'val' data
del f['02801938']['val'] # Delete the old 'val' point cloud data
f['02801938'].create_dataset('val', data=val_XYZRGB_data) # Add the new 'val' point cloud data
print(list(hospital_XYZRGB_dataset.keys()))
# (3) for 'test' data
del f['02801938']['test'] # Delete the old 'test' point cloud data
f['02801938'].create_dataset('test', data=val_XYZRGB_data) # Add the new 'test' point cloud data
print(list(hospital_XYZRGB_dataset.keys()))


# endregion
