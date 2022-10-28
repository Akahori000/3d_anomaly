import h5py
import numpy as np
import pandas as pd
import glob
import json
import os
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import copy
import random
import torch
from torch.utils import data

point_num = 2048

# csv または　pcdを読み込んで、multi-viewpointcloudをつくり、h5に突っ込む

# lamp.h5などを作る
def make_h5py(points, angle, h5_file_path, h5_file_name):
    path = h5_file_path
    name = h5_file_name + str(int(angle)).zfill(3) # 文字列右うめ

    lbl = h5_file_name
    arr = points
    with h5py.File(path + name +'.h5', 'a') as f:
        f.create_dataset('data', data=arr)
        f.create_dataset('label', data=lbl)
        f.close()

def make_h5py_all(h5_file_path, h5_file_name):
    dirs = [h5_file_path]
    dic = [h5_file_name]

    for dr in dirs:
        for i, d in enumerate(dic):
            l = glob.glob(dr + d +'*')
            arr = np.zeros((len(l), 2048, 3))
            lbl = np.full(len(l), i)
            print(d, lbl)

            for cnt, i in enumerate(l):
                print(cnt, i)
                with h5py.File(i, 'r') as f:
                    arr[cnt] = f['data']

            # if os.path.exists(dr + d +'.h5'):
            #     os.remove(dr + d +'.h5')

            with h5py.File(dr + d +'.h5', 'a') as f:
                f.create_dataset('data', data=arr)
                f.create_dataset('label', data=lbl)
                f.close()


def make_h5py_direct(pointarr, clsnumber, h5_file_path, h5_file_name): #class.h5みたいなのをnarrayから直接つくる clsnumberは同クラスで全部同じint
    dirs = [h5_file_path]
    dic = [h5_file_name]
    lbl = np.full(pointarr.shape[0], clsnumber)

    with h5py.File(dirs[0] + dic[0] +'.h5', 'a') as f:
        f.create_dataset('data', data=pointarr)
        f.create_dataset('label', data=lbl)
        f.close()

def save_pcd_image(pcd, image_path):
    # R = pcd.get_rotation_matrix_from_xyz((-0.2 * np.pi, -0.05 * np.pi, -0.6 * np.pi))
    # xyz = pcd.rotate(R, center=(0,0,0))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    file = image_path + ".png"
    vis.capture_screen_image(file)
    vis.destroy_window()

def imagesave_shape(coord, image_path, i):
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(coord)
    
    #o3d.visualization.draw_geometries([pcd1], window_name="angle" + str(i), width=640, height=480)
    save_pcd_image(pcd1, image_path)

def visualize_shape(coord):
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(coord)
    
    #o3d.visualization.draw_geometries([pcd1], window_name="pcd1", width=640, height=480)

def get_pcd_from_multi_angles(points, point_num, save_path, image_path, original_file_name, save_pcd_flg):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if save_pcd_flg == 1:
        o3d.io.write_point_cloud(save_path + "chair_angle.pcd", pcd)

    # なんかようわからんが、この計算で点群を球で囲ったときの直径が取れるらしい.....
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print("diameter", diameter)

    #npcd = np.array(points)

    angle_gap = 5
    angle_num = int(360/angle_gap)
    arr_pcd = np.zeros((angle_num, point_num, 3))

    for arrcnt, i in enumerate(range(0, 360, angle_gap)):
        print(arrcnt, 'angle:', i)
        cam_y = 0
        cam_x = diameter * np.cos(np.radians(i))
        cam_z = diameter * np.sin(np.radians(i)) # ここがまちがっていた…i×angle_gapになってた
        print(cam_x, cam_z)

        #camera = [0, diameter, diameter]
        camera = [cam_x, cam_y, cam_z]
        # 逆投影するための外円定義 and この変数自体は(by　武田修論)
        radius = diameter * 50

        # 単視点の点群取ってくる
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        pcd1 = pcd.select_by_index(pt_map)

        points_angle = np.array(pcd1.points)

        ipath = image_path + original_file_name + '_angle' + str(i) +'_points' + str(len(pcd1.points))
        imagesave_shape(points_angle, ipath, i)
        df = pd.DataFrame(points_angle)

        # 固定点群数にする
        point_idx = [j for j in range(points_angle.shape[0])]
        point_idx = np.array(point_idx)
        if points_angle.shape[0] >= point_num:
            point_choice = np.random.choice(point_idx, point_num, replace=False)
        else:
            point_choice = np.random.choice(point_idx, point_num, replace=True)
        points_angle = points_angle[point_choice, :]


        #make_h5py(points_angle, i, , h5_file_path, h5_file_name)
        #df = pd.DataFrame(points_angle)
        #df.to_csv(save_path + original_file_name + '_angle' + str(i) +'_points' + str(len(points_angle)) + '.csv')

        arr_pcd[arrcnt, :, :] = points_angle

    return(arr_pcd)

# #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-0, -0, -0])

# # 可視化#
#     #o3d.visualization.draw_geometries([pcd1, mesh_frame])
#     #print(i * angle_gap)
#     #deform_xyz.save_pcd_image_simple(pcd1, str(int(i * angle_gap)), 'images_angle/', [0, (i * angle_gap) -90 , 0])
#     #o3d.io.write_point_cloud('pcd_angles/' + str(int(i * angle_gap)) +".ply", pcd1)
# print(len(pcd1.points)) # images_downsampled_angles
# #pcd2 = adj_the_num_of_pcd(pcd1, 1738) # 1681
#     #o3d.visualization.draw_geometries([pcd2, mesh_frame])
# #deform_xyz.save_pcd_image_simple(pcd2, str((i * angle_gap)), 'image_angle2/down_', [0, (i * angle_gap) -90 , 0])
# #o3d.io.write_point_cloud('pcd_angles/down_' + str((i * angle_gap)) +".ply", pcd2)



# # csvを読み込み
# file_name = 'chair_104_points14434'
# original_file_name = 'chair_104'
# df = pd.read_csv('./data/pcd_all_csv/' + file_name + '.csv')
# points = df.to_numpy()[:,1:]
# print(points)
# points = points - np.mean(points, axis=0)
# #visualize_shape(points, 'test')

#読みこんだ点群をmulti-viewで回しながら画像とpoint_cloud.csvを保存
folder = './data/pcd_chair_anomalies/original/'
save_path = folder + 'multi_view_csv/'
image_path = folder + 'multi_view_images/'
original_file_name = 'chair104'
pcd = o3d.io.read_point_cloud(folder + original_file_name + '.pcd')
visualize_shape(pcd.points)
save_pcd_flg = 4
point_arr = get_pcd_from_multi_angles(pcd.points, point_num, save_path, image_path, original_file_name, save_pcd_flg)

#multi-viewのcsvのpoint_cloudをh5にする
h5_file_path = folder + 'multi_view_h5/'
clsnumber = 1 # ここは適宜変える　同クラスで同じ値
make_h5py_direct(point_arr, clsnumber, h5_file_path, original_file_name)
#make_h5py_all()

# 確認
# file = './data/train/airplane.h5'
# with h5py.File(file, 'r') as f:
#     print(f['data'][0], f['label'][0])

# file = './data/data_h5_angle/chair_angles.h5'
# with h5py.File(file, 'r') as f:
#     print(f['data'][0], f['label'][0])
