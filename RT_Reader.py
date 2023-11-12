import numpy as np
import pandas as pd
import torch
import pypose as pp


def PoseReader(seq, kittiPath, readpose):
    df = pd.read_table(kittiPath + f'{seq}/pose_left.txt', header=None, sep=' ') # poses
    if readpose:
        df.columns = ['t0', 't1', 't2', 'qx', 'qy', 'qz', 'qw']
    else:
        df.columns = ['t0', 't1', 't2', 'qx', 'qy', 'qz', 'qw', '0', '0', '0', '1', 'nan']
    process_index = range(0, len(df))
    poses = []
    for i in range(len(df)):
        TR = pp.SE3([df['t0'][i], df['t1'][i], df['t2'][i], df['qx'][i], df['qy'][i], df['qz'][i], df['qw'][i]])
        T = TR.matrix()
        poses.append(T)
    return poses, process_index