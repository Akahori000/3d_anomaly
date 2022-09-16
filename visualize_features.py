import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


from sklearn.manifold import TSNE

dir = './feature/c_e_epoc_100_data7119/'

features = pd.read_csv(dir + 'feature.csv')
names = pd.read_csv(dir + 'name.csv')
mu = pd.read_csv(dir + 'mu.csv')
var = pd.read_csv(dir + 'var.csv')

features = features.iloc[:,1:].values
names = names.iloc[:, 1:].values.ravel()
mu = mu.iloc[:, 1:].values
var = var.iloc[:, 1:].values

clsnm = np.zeros(7) # 各クラスのデータ数
ftr = np.zeros((features.shape)) # クラスごと並び替え
m = np.zeros((mu.shape))
v = np.zeros((var.shape))
for cls in range(7):
    idx = [i for i, x in enumerate(names.tolist()) if x == cls]
    ftr[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = features[idx, :]
    m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
    v[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = var[idx, :]
    clsnm[cls] = int(len(idx))

clsnm = np.array(clsnm, dtype=int)
num = np.array([0, clsnm[0], np.sum(clsnm[:2]), np.sum(clsnm[:3]), np.sum(clsnm[:4]), np.sum(clsnm[:5]), np.sum(clsnm[:6]), np.sum(clsnm)])

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


mu = pd.read_csv('mu_processed.csv')
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
plt.savefig(os.path.join(dir, "mu_tsne.png"))
plt.show()
plt.close()

var = pd.read_csv('var_processed.csv')
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