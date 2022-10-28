import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import umap.umap_ as umap

from sklearn.manifold import TSNE

traindir = './feature/model1_chair/c_epoc_299_data7119/'
testdir=  './feature/model1_chair/c_epoc_299_data2034/'


dir = './feature/model1_chair/c_epoc_299_all/'
if not os.path.exists(dir):
    os.makedirs(dir)

#trained_data
#features = pd.read_csv(traindir + 'feature.csv')
names = pd.read_csv(traindir + 'name.csv')
mu = pd.read_csv(traindir + 'mu.csv')
#var = pd.read_csv(traindir + 'var.csv')

#features = features.iloc[:,1:].values
names = names.iloc[:, 1:].values.ravel()
mu = mu.iloc[:, 1:].values
#var = var.iloc[:, 1:].values

clsnm = np.zeros(7) # 各クラスのデータ数
#ftr = np.zeros((features.shape)) # クラスごと並び替え
m = np.zeros((mu.shape))
#v = np.zeros((var.shape))
for cnt in range(7):
    idx = [i for i, x in enumerate(names.tolist()) if x == cnt] # 特定のクラスのindexを抽出
#    ftr[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = features[idx, :]
    m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
#    v[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = var[idx, :]
    clsnm[cnt] = int(len(idx))

clsnm = np.array(clsnm, dtype=int)
num = np.array([0, clsnm[0], np.sum(clsnm[:2]), np.sum(clsnm[:3]), np.sum(clsnm[:4]), np.sum(clsnm[:5]), np.sum(clsnm[:6]), np.sum(clsnm)])
y = np.zeros(num[7])
for i in range(7):
    y[num[i]:num[i+1]] = i

# test data
#features1 = pd.read_csv(testdir + 'feature.csv')
names1 = pd.read_csv(testdir + 'name.csv')
mu1 = pd.read_csv(testdir + 'mu.csv')
#var1 = pd.read_csv(testdir + 'var.csv')

#features1 = features1.iloc[:,1:].values
names1 = names1.iloc[:, 1:].values.ravel()
mu1 = mu1.iloc[:, 1:].values
#var1 = var1.iloc[:, 1:].values

clsnm1 = np.zeros(7) # 各クラスのデータ数
#ftr1 = np.zeros((features1.shape)) # クラスごと並び替え
m1 = np.zeros((mu1.shape))
#v1 = np.zeros((var1.shape))
for cnt in range(7):
    idx = [i for i, x in enumerate(names1.tolist()) if x == cnt] # 特定のクラスのindexを抽出
#    ftr1[int(np.sum(clsnm1)) : (int(np.sum(clsnm1) + len(idx))), :] = features1[idx, :]
    m1[int(np.sum(clsnm1)) : (int(np.sum(clsnm1) + len(idx))), :] = mu1[idx, :]
#    v1[int(np.sum(clsnm1)) : (int(np.sum(clsnm1) + len(idx))), :] = var1[idx, :]
    clsnm1[cnt] = int(len(idx))

clsnm1 = np.array(clsnm1, dtype=int)
num1 = np.array([0, clsnm1[0], np.sum(clsnm1[:2]), np.sum(clsnm1[:3]), np.sum(clsnm1[:4]), np.sum(clsnm1[:5]), np.sum(clsnm1[:6]), np.sum(clsnm1)])
y1 = np.zeros(num1[7])
for i in range(7):
    y1[num1[i]:num1[i+1]] = i+7


#ftr = np.vstack([ftr, ftr1])
names = np.hstack([names, names1])
m = np.vstack([m, m1])
#v = np.vstack([v, v1])

num1 = num1 + np.sum(clsnm)
num = np.hstack([num[:7], num1])
y = np.hstack([y, y1])
y = np.array(y, dtype = int)
print(num)


col = ['dodgerblue', 'green', 'purple', 'gray', 'yellow', 'pink', 'red','skyblue', 'yellowgreen', 'navy', 'black', 'orange', 'magenta', 'brown']
plabels = ['lamp', 'chair', 'table', 'car', 'sofa', 'rifle', 'airplane', 'lamp(test)', 'chair(test)', 'table(test)', 'car(test)', 'sofa(test)', 'rifle(test)', 'airplane(test)']

#---Umap----##
X= m

# 次元削減する
mapper = umap.UMAP(random_state=0)
embedding = mapper.fit_transform(X)

# 結果を二次元でプロットする
embedding_x = embedding[:, 0]
embedding_y = embedding[:, 1]
fig = plt.figure(figsize=(10.0, 10.0))
for n in np.unique(y):
    plt.scatter(embedding_x[y == n],
                embedding_y[y == n],
                s=1,
                color=col[int(n)],
                label=plabels[n])
# グラフを表示する
plt.grid()
plt.legend(fontsize = 6)
plt.savefig(os.path.join(dir, "mu_umap.png"))
#plt.show()
plt.close()


fit = umap.UMAP(n_components=3)
u = fit.fit_transform(X)
fig = plt.figure(figsize=(10.0, 10.0))
ax = fig.add_subplot(111, projection='3d')

for n in np.unique(y):
    ax.scatter(u[y==n][:,0], u[y==n][:,1], u[y==n][:,2], c=col[int(n)], s=1, label=plabels[n])
ax.legend(fontsize = 6)
plt.savefig(os.path.join(dir, "mu_umap_3d.png"))
plt.show()
plt.close()

##---TSNE---##

feat_reduced = TSNE(n_components=3).fit_transform(m)
fig3 = plt.figure(figsize=(10.0, 10.0))
ax = Axes3D(fig3)
ax.scatter(feat_reduced[num[0]:num[1], 0], feat_reduced[num[0]:num[1], 1], feat_reduced[num[0]:num[1], 2], marker='.', c='dodgerblue')  # lamp
ax.scatter(feat_reduced[num[1]:num[2], 0], feat_reduced[num[1]:num[2], 1], feat_reduced[num[1]:num[2], 2], marker='.', c='green')  # chair
ax.scatter(feat_reduced[num[2]:num[3], 0], feat_reduced[num[2]:num[3], 1], feat_reduced[num[2]:num[3], 2], marker='.', c='purple')  # table
ax.scatter(feat_reduced[num[3]:num[4], 0], feat_reduced[num[3]:num[4], 1], feat_reduced[num[3]:num[4], 2], marker='.', c='gray') # car
ax.scatter(feat_reduced[num[4]:num[5], 0], feat_reduced[num[4]:num[5], 1], feat_reduced[num[4]:num[5], 2], marker='.', c='yellow')  # sofa
ax.scatter(feat_reduced[num[5]:num[6], 0], feat_reduced[num[5]:num[6], 1], feat_reduced[num[5]:num[6], 2], marker='.', c='pink')  # rifle
ax.scatter(feat_reduced[num[6]:num[7], 0], feat_reduced[num[6]:num[7], 1], feat_reduced[num[6]:num[7], 2], marker='.', c='red')  # airplane

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

# feat_reduced = TSNE(n_components=3).fit_transform(ftr)
# fig3 = plt.figure(figsize=(10.0, 10.0))
# ax = Axes3D(fig3)
# ax.scatter(feat_reduced[num[0]:num[1], 0], feat_reduced[num[0]:num[1], 1], feat_reduced[num[0]:num[1], 2], marker='.', c='dodgerblue') 
# ax.scatter(feat_reduced[num[1]:num[2], 0], feat_reduced[num[1]:num[2], 1], feat_reduced[num[1]:num[2], 2], marker='.', c='green') 
# ax.scatter(feat_reduced[num[2]:num[3], 0], feat_reduced[num[2]:num[3], 1], feat_reduced[num[2]:num[3], 2], marker='.', c='purple') 
# ax.scatter(feat_reduced[num[3]:num[4], 0], feat_reduced[num[3]:num[4], 1], feat_reduced[num[3]:num[4], 2], marker='.', c='gray') 
# ax.scatter(feat_reduced[num[4]:num[5], 0], feat_reduced[num[4]:num[5], 1], feat_reduced[num[4]:num[5], 2], marker='.', c='yellow') 
# ax.scatter(feat_reduced[num[5]:num[6], 0], feat_reduced[num[5]:num[6], 1], feat_reduced[num[5]:num[6], 2], marker='.', c='pink') 
# ax.scatter(feat_reduced[num[6]:num[7], 0], feat_reduced[num[6]:num[7], 1], feat_reduced[num[6]:num[7], 2], marker='.', c='red') 

# ax.scatter(feat_reduced[num[7]:num[8], 0], feat_reduced[num[7]:num[8], 1], feat_reduced[num[7]:num[8], 2], marker='.', c='skyblue') 
# ax.scatter(feat_reduced[num[8]:num[9], 0], feat_reduced[num[8]:num[9], 1], feat_reduced[num[8]:num[9], 2], marker='.', c='yellowgreen') 
# ax.scatter(feat_reduced[num[9]:num[10], 0], feat_reduced[num[9]:num[10], 1], feat_reduced[num[9]:num[10], 2], marker='.', c='navy') 
# ax.scatter(feat_reduced[num[10]:num[11], 0], feat_reduced[num[10]:num[11], 1], feat_reduced[num[10]:num[11], 2], marker='.', c='black') 
# ax.scatter(feat_reduced[num[11]:num[12], 0], feat_reduced[num[11]:num[12], 1], feat_reduced[num[11]:num[12], 2], marker='.', c='orange') 
# ax.scatter(feat_reduced[num[12]:num[13], 0], feat_reduced[num[12]:num[13], 1], feat_reduced[num[12]:num[13], 2], marker='.', c='magenta') 
# ax.scatter(feat_reduced[num[13]:num[14], 0], feat_reduced[num[13]:num[14], 1], feat_reduced[num[13]:num[14], 2], marker='.', c='brown') 
# plt.savefig(os.path.join(dir, "feat_tsne.png"))
# plt.show()
# plt.close()



# feat_reduced = TSNE(n_components=3).fit_transform(v)
# fig3 = plt.figure(figsize=(10.0, 10.0))
# ax = Axes3D(fig3)
# ax.scatter(feat_reduced[num[0]:num[1], 0], feat_reduced[num[0]:num[1], 1], feat_reduced[num[0]:num[1], 2], marker='.', c='dodgerblue') 
# ax.scatter(feat_reduced[num[1]:num[2], 0], feat_reduced[num[1]:num[2], 1], feat_reduced[num[1]:num[2], 2], marker='.', c='green') 
# ax.scatter(feat_reduced[num[2]:num[3], 0], feat_reduced[num[2]:num[3], 1], feat_reduced[num[2]:num[3], 2], marker='.', c='purple') 
# ax.scatter(feat_reduced[num[3]:num[4], 0], feat_reduced[num[3]:num[4], 1], feat_reduced[num[3]:num[4], 2], marker='.', c='gray') 
# ax.scatter(feat_reduced[num[4]:num[5], 0], feat_reduced[num[4]:num[5], 1], feat_reduced[num[4]:num[5], 2], marker='.', c='yellow') 
# ax.scatter(feat_reduced[num[5]:num[6], 0], feat_reduced[num[5]:num[6], 1], feat_reduced[num[5]:num[6], 2], marker='.', c='pink') 
# ax.scatter(feat_reduced[num[6]:num[7], 0], feat_reduced[num[6]:num[7], 1], feat_reduced[num[6]:num[7], 2], marker='.', c='red') 

# ax.scatter(feat_reduced[num[7]:num[8], 0], feat_reduced[num[7]:num[8], 1], feat_reduced[num[7]:num[8], 2], marker='.', c='skyblue') 
# ax.scatter(feat_reduced[num[8]:num[9], 0], feat_reduced[num[8]:num[9], 1], feat_reduced[num[8]:num[9], 2], marker='.', c='yellowgreen') 
# ax.scatter(feat_reduced[num[9]:num[10], 0], feat_reduced[num[9]:num[10], 1], feat_reduced[num[9]:num[10], 2], marker='.', c='navy') 
# ax.scatter(feat_reduced[num[10]:num[11], 0], feat_reduced[num[10]:num[11], 1], feat_reduced[num[10]:num[11], 2], marker='.', c='black') 
# ax.scatter(feat_reduced[num[11]:num[12], 0], feat_reduced[num[11]:num[12], 1], feat_reduced[num[11]:num[12], 2], marker='.', c='orange') 
# ax.scatter(feat_reduced[num[12]:num[13], 0], feat_reduced[num[12]:num[13], 1], feat_reduced[num[12]:num[13], 2], marker='.', c='magenta') 
# ax.scatter(feat_reduced[num[13]:num[14], 0], feat_reduced[num[13]:num[14], 1], feat_reduced[num[13]:num[14], 2], marker='.', c='brown') 
# plt.savefig(os.path.join(dir, "var_tsne.png"))
# plt.show()
# plt.close()

