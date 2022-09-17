import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


from sklearn.manifold import TSNE

traindir = './feature/c_e_epoc_299_data7119/'
testdir=  './feature/c_e_epoc_299_data2034/'

dir = './feature/c_e_epoc_299_all/'
if not os.path.exists(dir):
    os.makedirs(dir)

#trained_data
features = pd.read_csv(traindir + 'feature.csv')
names = pd.read_csv(traindir + 'name.csv')
mu = pd.read_csv(traindir + 'mu.csv')
var = pd.read_csv(traindir + 'var.csv')

features = features.iloc[:,1:].values
names = names.iloc[:, 1:].values.ravel()
mu = mu.iloc[:, 1:].values
var = var.iloc[:, 1:].values

clsnm = np.zeros(7) # 各クラスのデータ数
ftr = np.zeros((features.shape)) # クラスごと並び替え
m = np.zeros((mu.shape))
v = np.zeros((var.shape))
for cls in range(7):
    idx = [i for i, x in enumerate(names.tolist()) if x == cls] # 特定のクラスのindexを抽出
    ftr[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = features[idx, :]
    m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
    v[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = var[idx, :]
    clsnm[cls] = int(len(idx))

clsnm = np.array(clsnm, dtype=int)
num = np.array([0, clsnm[0], np.sum(clsnm[:2]), np.sum(clsnm[:3]), np.sum(clsnm[:4]), np.sum(clsnm[:5]), np.sum(clsnm[:6]), np.sum(clsnm)])


# test data
features1 = pd.read_csv(testdir + 'feature.csv')
names1 = pd.read_csv(testdir + 'name.csv')
mu1 = pd.read_csv(testdir + 'mu.csv')
var1 = pd.read_csv(testdir + 'var.csv')

features1 = features1.iloc[:,1:].values
names1 = names1.iloc[:, 1:].values.ravel()
mu1 = mu1.iloc[:, 1:].values
var1 = var1.iloc[:, 1:].values

clsnm1 = np.zeros(7) # 各クラスのデータ数
ftr1 = np.zeros((features1.shape)) # クラスごと並び替え
m1 = np.zeros((mu1.shape))
v1 = np.zeros((var1.shape))
for cls in range(7):
    idx = [i for i, x in enumerate(names1.tolist()) if x == cls] # 特定のクラスのindexを抽出
    ftr1[int(np.sum(clsnm1)) : (int(np.sum(clsnm1) + len(idx))), :] = features1[idx, :]
    m1[int(np.sum(clsnm1)) : (int(np.sum(clsnm1) + len(idx))), :] = mu1[idx, :]
    v1[int(np.sum(clsnm1)) : (int(np.sum(clsnm1) + len(idx))), :] = var1[idx, :]
    clsnm1[cls] = int(len(idx))

clsnm1 = np.array(clsnm1, dtype=int)
num1 = np.array([0, clsnm1[0], np.sum(clsnm1[:2]), np.sum(clsnm1[:3]), np.sum(clsnm1[:4]), np.sum(clsnm1[:5]), np.sum(clsnm1[:6]), np.sum(clsnm1)])



ftr = np.vstack([ftr, ftr1])
#names = np.hstack([names, names1])
m = np.vstack([m, m1])
v = np.vstack([v, v1])

num1 = num1 + np.sum(clsnm)
num = np.hstack([num, num1])




feat_reduced = TSNE(n_components=3).fit_transform(m)
fig3 = plt.figure(figsize=(10.0, 10.0))
ax = Axes3D(fig3)
ax.scatter(feat_reduced[num[0]:num[1], 0], feat_reduced[num[0]:num[1], 1], feat_reduced[num[0]:num[1], 2], marker='.', c='blue') 
ax.scatter(feat_reduced[num[1]:num[2], 0], feat_reduced[num[1]:num[2], 1], feat_reduced[num[1]:num[2], 2], marker='.', c='green') 
ax.scatter(feat_reduced[num[2]:num[3], 0], feat_reduced[num[2]:num[3], 1], feat_reduced[num[2]:num[3], 2], marker='.', c='purple') 
ax.scatter(feat_reduced[num[3]:num[4], 0], feat_reduced[num[3]:num[4], 1], feat_reduced[num[3]:num[4], 2], marker='.', c='gray') 
ax.scatter(feat_reduced[num[4]:num[5], 0], feat_reduced[num[4]:num[5], 1], feat_reduced[num[4]:num[5], 2], marker='.', c='yellow') 
ax.scatter(feat_reduced[num[5]:num[6], 0], feat_reduced[num[5]:num[6], 1], feat_reduced[num[5]:num[6], 2], marker='.', c='pink') 
ax.scatter(feat_reduced[num[6]:num[7], 0], feat_reduced[num[6]:num[7], 1], feat_reduced[num[6]:num[7], 2], marker='.', c='red') 

ax.scatter(feat_reduced[num[7]:num[8], 0], feat_reduced[num[7]:num[8], 1], feat_reduced[num[7]:num[8], 2], marker='.', c='skyblue') 
ax.scatter(feat_reduced[num[8]:num[9], 0], feat_reduced[num[8]:num[9], 1], feat_reduced[num[8]:num[9], 2], marker='.', c='yellowgreen') 
ax.scatter(feat_reduced[num[9]:num[10], 0], feat_reduced[num[9]:num[10], 1], feat_reduced[num[9]:num[10], 2], marker='.', c='navy') 
ax.scatter(feat_reduced[num[10]:num[11], 0], feat_reduced[num[10]:num[11], 1], feat_reduced[num[10]:num[11], 2], marker='.', c='black') 
ax.scatter(feat_reduced[num[11]:num[12], 0], feat_reduced[num[11]:num[12], 1], feat_reduced[num[11]:num[12], 2], marker='.', c='orange') 
ax.scatter(feat_reduced[num[12]:num[13], 0], feat_reduced[num[12]:num[13], 1], feat_reduced[num[12]:num[13], 2], marker='.', c='magenta') 
ax.scatter(feat_reduced[num[13]:num[14], 0], feat_reduced[num[13]:num[14], 1], feat_reduced[num[13]:num[14], 2], marker='.', c='brown') 
plt.savefig(os.path.join(dir, "mu_tsne.png"))
plt.show()
plt.close()



feat_reduced = TSNE(n_components=3).fit_transform(ftr)
fig3 = plt.figure(figsize=(10.0, 10.0))
ax = Axes3D(fig3)
ax.scatter(feat_reduced[num[0]:num[1], 0], feat_reduced[num[0]:num[1], 1], feat_reduced[num[0]:num[1], 2], marker='.', c='blue') 
ax.scatter(feat_reduced[num[1]:num[2], 0], feat_reduced[num[1]:num[2], 1], feat_reduced[num[1]:num[2], 2], marker='.', c='green') 
ax.scatter(feat_reduced[num[2]:num[3], 0], feat_reduced[num[2]:num[3], 1], feat_reduced[num[2]:num[3], 2], marker='.', c='purple') 
ax.scatter(feat_reduced[num[3]:num[4], 0], feat_reduced[num[3]:num[4], 1], feat_reduced[num[3]:num[4], 2], marker='.', c='gray') 
ax.scatter(feat_reduced[num[4]:num[5], 0], feat_reduced[num[4]:num[5], 1], feat_reduced[num[4]:num[5], 2], marker='.', c='yellow') 
ax.scatter(feat_reduced[num[5]:num[6], 0], feat_reduced[num[5]:num[6], 1], feat_reduced[num[5]:num[6], 2], marker='.', c='pink') 
ax.scatter(feat_reduced[num[6]:num[7], 0], feat_reduced[num[6]:num[7], 1], feat_reduced[num[6]:num[7], 2], marker='.', c='red') 
plt.savefig(os.path.join(dir, "feature_tsne.png"))
plt.show()
plt.close()

feat_reduced = TSNE(n_components=3).fit_transform(v)
fig3 = plt.figure(figsize=(10.0, 10.0))
ax = Axes3D(fig3)
ax.scatter(feat_reduced[num[0]:num[1], 0], feat_reduced[num[0]:num[1], 1], feat_reduced[num[0]:num[1], 2], marker='.', c='blue') 
ax.scatter(feat_reduced[num[1]:num[2], 0], feat_reduced[num[1]:num[2], 1], feat_reduced[num[1]:num[2], 2], marker='.', c='green') 
ax.scatter(feat_reduced[num[2]:num[3], 0], feat_reduced[num[2]:num[3], 1], feat_reduced[num[2]:num[3], 2], marker='.', c='purple') 
ax.scatter(feat_reduced[num[3]:num[4], 0], feat_reduced[num[3]:num[4], 1], feat_reduced[num[3]:num[4], 2], marker='.', c='gray') 
ax.scatter(feat_reduced[num[4]:num[5], 0], feat_reduced[num[4]:num[5], 1], feat_reduced[num[4]:num[5], 2], marker='.', c='yellow') 
ax.scatter(feat_reduced[num[5]:num[6], 0], feat_reduced[num[5]:num[6], 1], feat_reduced[num[5]:num[6], 2], marker='.', c='pink') 
ax.scatter(feat_reduced[num[6]:num[7], 0], feat_reduced[num[6]:num[7], 1], feat_reduced[num[6]:num[7], 2], marker='.', c='red') 
plt.savefig(os.path.join(dir, "var_tsne.png"))
plt.show()
plt.close()