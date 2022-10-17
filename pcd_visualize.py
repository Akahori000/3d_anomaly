import h5py
import numpy as np
import pandas as pd
import glob
import json
import os
#import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import copy
import random
import torch
from torch.utils import data


def load_h5py(path):
    all_data = []
    all_label = []
    for h5_name in path:
        # 読み込み
        f = h5py.File(h5_name, "r+")
        data = f["data"][:].astype("float32")
        label = f["label"][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
        #print(f"{label[0]} : {len(data)}")
    return all_data, all_label

def load_h5py_onebyone(path):
    all_data = []
    all_label = []

    f = h5py.File(path, "r+")
    data = f["data"][:].astype("float32")
    #label = f["label"][:]
    f.close()
    all_data.append(data)
    #all_label.append(label)
        #print(f"{label[0]} : {len(data)}")
    return all_data, all_label


names = ['lamp', 'chair', 'table', 'car', 'sofa', 'rifle', 'airplane']

def scatter3d(_x, cls, i):

    df = pd.DataFrame(_x)
    df.to_csv('./data/pcd_all_csv/' + cls + '_' + str(i) + '_points'+ str(len(_x)) +".csv")

    fig = plt.figure(figsize=(10.0, 10.0))
    ax1 = fig.add_subplot(111, projection="3d")
    sc = ax1.scatter(_x[:, 0], _x[:, 1], _x[:, 2])
    #ax1.set_title(label=names[cls]+str(i)+'  points:' + len(_X))
    #fig.savefig("./data/point_clouds_all_random/" + names[cls] + '_' + str(i) + '_points'+ str(len(_x)) + ".png")

    ax1.set_title(label=cls+str(i)+'  points:' + str(len(_x)))
    fig.savefig("./data/point_clouds_all_random/" + cls + '_' + str(i) + '_points'+ str(len(_x)) + ".png")
    
    #plt.show()
    plt.close()
    plt.clf()



# path = './data/test_randomselect/'
# l = glob.glob(path + '*.h5')
# for i, nm in enumerate(l):
#     dt, lb = load_h5py_onebyone(nm)
#     scatter3d(np.array(dt).reshape((2048, 3)), 1, i)

def loadOBJ(filepath: str) -> list:

    vertices = []
    if os.path.exists(filepath):
        file = open(filepath, "r")
        for line in file:
            vals = line.split()

            if len(vals) == 0:
                continue

            if vals[0] == "v":
                v = list(map(float, vals[1:4]))
                vertices.append(v)
    return vertices

def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point)
    """
    return ((x - y) ** 2).sum(axis=1)


def fartherst_point_sampling(
    points: np.ndarray,
    num_sample_point: int,
    initial_idx=None,
    metrics=l2_norm,
    indices_dtype=np.int32,
    distances_dtype=np.float32,
) -> np.ndarray:
    assert points.ndim == 2, "input points shoud be 2-dim array (n_points, coord_dim)"

    num_point, coord_dim = points.shape
    indices = np.zeros((num_sample_point,), dtype=indices_dtype)
    distances = np.zeros((num_sample_point, num_point), dtype=distances_dtype)

    if initial_idx is None:
        indices[0] = np.random.randint(len(points))
    else:
        indices[0] = initial_idx

    farthest_point = points[indices[0]]

    min_distances = metrics(farthest_point[None, :], points)
    distances[0, :] = min_distances
    for i in range(1, num_sample_point):
        indices[i] = np.argmax(min_distances, axis=0)
        farthest_point = points[indices[i]]
        dist = metrics(farthest_point[None, :], points)
        distances[i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices


class ShapeNetDataset(data.Dataset):
    def __init__(self, csv_file, sampling="fps", n_point=2000):
        super().__init__()
        assert sampling.lower() in [
            "fps",
            "random",
            "order",
        ], "The sampling method must be 'fps','random', or 'order'!"

        self.df = pd.read_csv(csv_file)
        self.n_point = n_point
        self.sampling = sampling

    def __len__(self):
        return len(self.df)

    def __checkpcds__(self, idx):
        label = self.df.iloc[idx]["label"]
        path = self.df.iloc[idx]["path"]
        point = loadOBJ(path)
        point = np.array(point)
        return point, label

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        point = loadOBJ(path)
        point = np.array(point)

        # vis_point = torch.tensor(point)
        # vis_points_3d(vis_point, f"./{idx}_original.png")

        point_idx = [i for i in range(point.shape[0])]
        point_idx = np.array(point_idx)

        # points sampling
        if self.sampling == "fps":
            point_idx = fartherst_point_sampling(point, self.n_point)
            point = point[point_idx]

        elif self.sampling == "random":
            if point.shape[0] >= self.n_point:
                point_choice = np.random.choice(point_idx, self.n_point, replace=False)
            else:
                point_choice = np.random.choice(point_idx, self.n_point, replace=True)
            point = point[point_choice, :]

        elif self.sampling == "order":
            point = point[: self.n_point]

        # point = torch.tensor(point)
        # vis_points_3d(point, f"./{idx}_sampling.png")

        label = self.df.iloc[idx]["label"]

        sample = {
            "data": point,
            "label": label,
            "name": path[26:26+28], # 名前の一部をとる(桁がばらばらなので…)
            "path": path,
        }

        return sample

def make_h5py_eachdata(csvpath):
    # if i == 0:
    #     csvpath = 'data/train.csv'
    #     savepath = 'data/train/'
    # elif i == 1:
    #     csvpath = 'data/test.csv'
    #     savepath = 'data/test/'
    # else:
    #     csvpath = 'data/val.csv'
    #     savepath = 'data/val/'

    print(os.getcwd())
    test = ShapeNetDataset(
        csv_file=csvpath,
        sampling="random",
        n_point=2048
    )

    dat = pd.read_csv(csvpath)
    print(dat, dat.info())

    cnt = 0
    for j in range(len(dat)):
        points, label = test.__checkpcds__(j)
        if points != []:
            cnt += 1
            print(label, len(points))
            scatter3d(points, label, cnt)
        # #print(dic)

        # #print(type(pcds))

        # with h5py.File(savepath + lb + '_' + nm + '.h5', 'a') as f:
        #     f.create_dataset('data', data=pcds)
        #     f.create_dataset('label', data=lb)
        #     f.create_dataset('name', data=nm)

        #     f.close()
        

make_h5py_eachdata('data/train.csv')




# path = ['data/train_random/lamp.h5', 'data/train_random/chair.h5', 'data/train_random/table.h5', 'data/train_random/car.h5', 'data/train_random/sofa.h5', 'data/train_random/rifle.h5', 'data/train_random/airplane.h5']
# normal_data, normal_name = load_h5py(path)
# print(np.unique(normal_name[0]))

# #df = pd.DataFrame(normal_data[0])
# #pcd1 = o3d.geometry.PointCloud()
# #pcd1.points = o3d.utility.Vector3dVector(pcd)
# for j in range(len(path)):
#     for i in range(10):
#         pcd = normal_data[j]
#         scatter3d(pcd[i,:,:], j, i)