'''
This demo code is originated from here "https://github.com/cattaneod/CMRNet/blob/master/preprocess/kitti_maps.py"
'''

import argparse
import os
import numpy as np
import pypose as pp
import open3d as o3
import torch
from tqdm import tqdm
import RT_Reader
import RGBD2PointCloud as RGBD2PtsCloud
from loadCalibration import loadCalibrationRigid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='000',
                        help='sequence')
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
    parser.add_argument('--start', default=538, help='Starting Frame')
    parser.add_argument('--end', default=562, help='End Frame')
    parser.add_argument('--map', default=None, help='Use map file')
    parser.add_argument('--remove_ground', default=False, help='Remove the ground when making global map')
    parser.add_argument('--create_submap', default=True, help='Create submaps from global map')
    parser.add_argument('--save_pcd', default=False, help='Save submaps in pcd format')
    parser.add_argument('--tartanPath', default='/home/jared/Large_datasets/data/P', help='Folder of the TartanAir dataset')
    parser.add_argument('--SubmapPath', default='/home/jared/Large_datasets/data/P', help='Folder of the Submap')

    args = parser.parse_args()
    print("Sequnce: ", args.sequence)
    map_file = args.map
    # read cam0_to_world
    poses, process_index = RT_Reader.PoseReader(args.sequence, args.tartanPath, readpose=True)
    process_index = list(process_index)

    first_frame = int(args.start)
    last_frame = min(len(poses), int(args.end))
    if args.remove_ground:
        f = open(f"./failed_frames_{args.sequence}.txt", "w")
    if map_file is None:
        pcl = o3.geometry.PointCloud()
        for i in tqdm(range(first_frame,last_frame)):
            frame = i
            rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(int(frame))
            pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
            pc_xyz = np.array(pcd_dense.points)
            pc_rgb = np.array(pcd_dense.colors)
            pc_ori = np.concatenate((pc_xyz, pc_rgb), axis=1)
            pc = pc_ori

            pc_rot = torch.tensor(pc[:, :4].copy()).to(args.device).to(torch.float)
            pc_rot[:, 3] = 1.
            # RT = poses[i].numpy()
            cam2world = poses[i]
            pose = cam2world.to(args.device).to(torch.float)
            # convert to torch.float
            pose = pose
            T_EDN2NED = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]], dtype=torch.float).to(args.device)
            pose =  pose @ T_EDN2NED.inverse()
            pc_rot = pose @ pc_rot.T
            pc_rot = pc_rot.T.clone().cpu() #.astype(np.float)
            pc_rot_points = np.array(pc_rot[:, :3])

            pcl_local = o3.geometry.PointCloud()
            pcl_local.points = o3.utility.Vector3dVector(pc_rot_points)
            pcl_local.colors = o3.utility.Vector3dVector(np.vstack((pc[:, 3].copy(), pc[:, 4].copy(), pc[:, 5].copy())).T)

            downpcd = pcl_local.voxel_down_sample(voxel_size=args.voxel_size)
            pcl.points.extend(downpcd.points)
            pcl.colors.extend(downpcd.colors)


        # downpcd_full = pcl.voxel_down_sample(voxel_size=args.voxel_size)
        downpcd_full = pcl
        downpcd, ind = downpcd_full.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)  #nb_neighbors=40, std_ratio=0.3
        o3.visualization.draw_geometries([downpcd])
        o3.io.write_point_cloud(f'./map-{args.sequence}_{args.voxel_size}_{first_frame}-{last_frame}.pcd', downpcd, write_ascii=True)
        print('map saved!')
    else:
        downpcd = o3.io.read_point_cloud(map_file)

    if args.create_submap:
        voxelized = torch.tensor(downpcd.points, dtype=torch.float64)
        voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float64)), 1)
        voxelized = voxelized.t()
        voxelized = voxelized.to(args.device)
        vox_intensity = torch.tensor(downpcd.colors, dtype=torch.float64)[:, 0:1].t()
        vox_intensity = vox_intensity.to(args.device)

        # SAVE SINGLE PCs
        bin_file_dir = os.path.join(args.SubmapPath, f'{args.sequence}','pc')
        if not os.path.exists(bin_file_dir):
            os.mkdir(bin_file_dir)
        if args.save_pcd:
            pcd_file_dir = os.path.join(args.SubmapPath, f's{args.sequence}_pcd')
            if not os.path.exists(pcd_file_dir):
                os.mkdir(pcd_file_dir)
        for i in tqdm(range(first_frame, last_frame)):
            cam2world = poses[i]
            pose = cam2world
            pose = pose.to(args.device)
            pose = pose.inverse()
            pose = pose.to(args.device)

            local_map = voxelized.clone()
            local_intensity = vox_intensity.clone()
            # local_map = local_map.to('cpu')
            local_map = torch.mm(pose, local_map).t()
            # range set
            indexes = local_map[:, 1] > -20.  # 25
            indexes = indexes & (local_map[:, 1] < 20.)  # 25
            indexes = indexes & (local_map[:, 0] > -20.)  # -10
            indexes = indexes & (local_map[:, 0] < 20.)  # 100
            local_map = local_map[indexes]
            local_intensity = local_intensity[:, indexes]
            local_map = local_map[:, :3]
            local_intensity = local_intensity.to('cpu')
            local_intensity = np.vstack((local_intensity, local_intensity, local_intensity)).T
            # submap save in pcd
            if args.save_pcd:
                pcd_nosam = o3.geometry.PointCloud()
                pcd_nosam.points = o3.utility.Vector3dVector(local_map)
                pcd_nosam.colors = o3.utility.Vector3dVector(local_intensity)
                o3.io.write_point_cloud(pcd_file_dir + f'/{process_index[i]:010d}.pcd', pcd_nosam)
            # save in bin
            downpcd_points = np.array(local_map.cpu(), dtype=np.float32)
            downpcd_color = local_intensity.astype(np.float32)

            np_x = downpcd_points[:, 0]
            np_y = downpcd_points[:, 1]
            np_z = downpcd_points[:, 2]
            np_i = downpcd_color[:, 0]

            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
            bin_file_path = os.path.join(bin_file_dir, f'{process_index[i]:010d}.bin')
            points_32.tofile(bin_file_path)




