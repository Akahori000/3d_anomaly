import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve

from ksm_example import ksm_exp as ksm

CLASS_NUM = 7
<<<<<<< HEAD
Anomaly_object = 'airplane' # ←ここでobject指定すること!
dictionary_dir =  './feature/model1_'+ Anomaly_object + '/c_epoc_299_data7119/'
test_dir = './feature/model1_'+ Anomaly_object + '/c_epoc_299_data2034/'
knl_dir = test_dir + 'kernel_pred2/'

dic_names = {'1622': 'lamp', '1323': 'chair', '1078':'table', '490':'car', '890':'sofa', '709': 'rifle', '1007': 'airplane'}
dic_names_test = {'464': 'lamp', '378': 'chair', '308':'table', '140':'car', '254':'sofa', '202': 'rifle', '288': 'airplane'}
=======
#dictionary_dir =  './feature/model1_airplane/c_k_epoc_299_data7119/'
#test_dir = './feature/model1_airplane/c_k_epoc_299_data2034/'

dictionary_dir =  './data/calculated_features/model1_rifle/both_features/c_epoc_299_data7119/'
test_dir = './data/calculated_features/model1_rifle/both_features/c_epoc_299_data2034/'

>>>>>>> 5e841a0e65d86df8c95ebe7eba9258044f65928c

if not os.path.exists(test_dir + 'pred/'):
    os.makedirs(test_dir + 'pred/')
if not os.path.exists(knl_dir):
    os.makedirs(knl_dir)

# 部分空間の基底計算
def calc_subspace(X, subdim): # Xは各列が１つのデータ
    v, sig, vt = np.linalg.svd(X)
    v = v[:, :subdim]
    return(v)


# 類似度計算 部分空間同士の比較
def calc_similarity(ds1, ds2):
    if np.all(ds1 == ds2):
        dissim = 0
        w = 0
    else:
        C = ds1.T @ ds2 @ ds2.T @ ds1
        w, v = np.linalg.eigh(C)                # w　= cos^2θ(canonical angles)
        dissim = np.mean(w)                 # canonical anlge = 0のとき、 cosθ = 1 なので, dissim（非類似度）は同じ同士では0

    #print('d_w',w)
    return(dissim)

# 入力ベクトルと部分空間の各基底との内積をとりノルムを計算
def calc_sim_vec_and_basis(input_feature, basis_vector):
  norm = np.linalg.norm(input_feature.T @ basis_vector, axis=0)   # ベクトルのノルムの大きさを計算する
  ave = np.mean(norm)
  return(ave)

def cos_sim(input_feature, basis_vector):
    sim = np.zeros(basis_vector.shape[1])
    for i in range(basis_vector.shape[1]):
        sim[i] = (np.dot(input_feature.T, basis_vector[:,i]) / (np.linalg.norm(input_feature) * np.linalg.norm(basis_vector[:,i])))**2      # cos^2θ マイナスなってはいけないので
    sim_ave = np.average(sim)
    return(sim_ave)

# 類似度計算 部分空間同士の比較 これはds1がデータ複数ないと使えない
def calc_sim2(ds1, ds2):
    if np.all(ds1 == ds2):
        dissim = 0
        w = 0
    else:
        w = np.zeros(ds2.shape[1])
        for i in range(ds2.shape[1]):
            C = ds1.T @ ds2[i] @ ds2[i].T @ ds1
            w[i], v = np.linalg.eigh(C)                # w　= cos^2θ(canonical angles)
        
        dissim = np.mean(w)                 # canonical anlge = 0のとき、 cosθ = 1 なので, dissim（非類似度）は同じ同士では0

    #print('d_w',w)
    return(dissim)

# A, B がそれぞれ基底行列 　(A.T@B) の特異値分解の特異値⇒正準角のcos 　
# dirを読み込んでfeature, mu, varをクラス毎(numに情報あり)に並び替えて返す関数
def get_features_and_sort(dir):
    features = pd.read_csv(dir + 'feature.csv')
    names = pd.read_csv(dir + 'name.csv')
    mu = pd.read_csv(dir + 'mu.csv')
    var = pd.read_csv(dir + 'var.csv')

    features = features.iloc[:,1:].values
    names = names.iloc[:, 1:].values.ravel()
    mu = mu.iloc[:, 1:].values
    var = var.iloc[:, 1:].values

    clsnm = np.zeros(CLASS_NUM) # 各クラスのデータ数
    ftr = np.zeros((features.shape)) # クラスごと並び替え
    m = np.zeros((mu.shape))
    v = np.zeros((var.shape))
    y = np.zeros(features.shape[0])
    label = np.zeros(features.shape[0]) #normal = 0, abnormal(anomaly) = 1 

    for cnt in range(CLASS_NUM):
        idx = [i for i, x in enumerate(names.tolist()) if x == cnt] # 特定のクラスのindexを抽出
        ftr[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = features[idx, :]
        m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
        v[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = var[idx, :]
        y[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx)))] = cnt
        clsnm[cnt] = int(len(idx))

    clsnm = np.array(clsnm, dtype=int)
    num = np.array([0, clsnm[0], np.sum(clsnm[:2]), np.sum(clsnm[:3]), np.sum(clsnm[:4]), np.sum(clsnm[:5]), np.sum(clsnm[:6]), np.sum(clsnm)])
    label[num[6]: num[7]] = 1

    return ftr, m, v, clsnm, num, y   # feature, mu, var, y 


def set_anomaly_labels(ftr, clsnm, num, anomaly_obj):

    anomaly_labels = np.zeros(len(ftr))

    anomaly_num = [k for k, v in dic_names_test.items() if v == anomaly_obj]
    anomaly_num = int(anomaly_num[0])
    
    for i in range(len(clsnm)):
        if clsnm[i] == anomaly_num:
            anomaly_labels[num[i]: num[i+1]] = 1 

    return anomaly_labels


def exclude_anomaly_obj(ftr, clsnm, num, anomaly_obj):

    # object_name → data_size取り出し
    anomaly_num = [k for k, v in dic_names.items() if v == anomaly_obj]
    anomaly_num = int(anomaly_num[0])

    # 
    ftre = np.zeros((ftr.shape[0] - anomaly_num, ftr.shape[1]))
    for i in range(len(num) - 1):
        if (num[i + 1] - num[i]) == anomaly_num:
            ftre = np.vstack((ftr[:num[i]], ftr[num[i+1]:]))
            clsnm = np.delete(clsnm, i)
    
    num = np.array([0, clsnm[0], np.sum(clsnm[:2]), np.sum(clsnm[:3]), np.sum(clsnm[:4]), np.sum(clsnm[:5]), np.sum(clsnm)])

    #print('exclude anomaly_obj (train)', anomaly_obj, anomaly_num, len(ftr)-len(ftre))

    return ftre, clsnm, num


def linear_subspace_test():
    # テスト開始
    sub_dim = 40
    ftr, mu, var, num, _, _ = get_features_and_sort(dictionary_dir)
    subdim_result = np.zeros((sub_dim, 7)) # subdim, accuracy0~2, roc0~2,

    # 部分空間の次元を変化させてテスト
    for subdim in (n+1 for n in range(sub_dim)):
        # 辞書データ
        dic_ftr_bases = np.zeros((CLASS_NUM, ftr.shape[1], subdim))
        dic_m_bases = np.zeros((CLASS_NUM, ftr.shape[1], subdim))
        dic_v_bases = np.zeros((CLASS_NUM, ftr.shape[1], subdim))

        for i in range (CLASS_NUM):
            dt = ftr[num[i]:num[i+1], :].T
            dic_ftr_bases[i] = calc_subspace(ftr[num[i]:num[i+1], :].T, subdim)
            dic_m_bases[i] = calc_subspace(mu[num[i]:num[i+1], :].T, subdim)
            dic_v_bases[i] = calc_subspace(var[num[i]:num[i+1], :].T, subdim)


        # テストデータ
        ftr, mu, var, num, y, label = get_features_and_sort(test_dir)
        # test_ftr_bases = np.zeros((CLASS_NUM, ftr.shape[1], subdim))
        # test_m_bases = np.zeros((CLASS_NUM, ftr.shape[1], subdim))
        # test_v_bases = np.zeros((CLASS_NUM, ftr.shape[1], subdim))


        # 各テストデータで行う
        testdt_num = ftr.shape[0]
        ftr_similarity = np.zeros((testdt_num, CLASS_NUM))
        mu_similarity =  np.zeros((testdt_num, CLASS_NUM))
        var_similarity = np.zeros((testdt_num, CLASS_NUM))

        ftr_prediction = np.zeros(testdt_num)
        mu_prediction = np.zeros(testdt_num)
        var_prediction = np.zeros(testdt_num)

        ftr_score = np.zeros(testdt_num)
        mu_score = np.zeros(testdt_num)
        var_score = np.zeros(testdt_num)

        for i in range(testdt_num): # 全データ
            for j in range(CLASS_NUM): # 各クラス
                ftr_similarity[i, j] = cos_sim((ftr[i, :].T).reshape(ftr.shape[1], 1), dic_ftr_bases[j])
                mu_similarity[i, j] = cos_sim((mu[i, :].T).reshape(ftr.shape[1], 1), dic_m_bases[j])
                var_similarity[i, j] = cos_sim((var[i, :].T).reshape(ftr.shape[1], 1), dic_v_bases[j])
            
            ftr_prediction[i] = int(np.argmax(ftr_similarity[i,:]))
            mu_prediction[i] = int(np.argmax(mu_similarity[i,:]))
            var_prediction[i] = int(np.argmax(var_similarity[i,:]))

            # 異常度の計算　 1- (テストデータと辞書空間のcos類似度の最大になるクラスのcos類似度)
            ftr_score[i] = 1 - max(ftr_similarity[i,:])
            mu_score[i] = 1 - max(mu_similarity[i,:])
            var_score[i] = 1 - max(var_similarity[i,:])

        # AUCの計算
        fpr, tpr, _ = roc_curve(label, ftr_score)
        ftr_roc_auc = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(label, mu_score)
        mu_roc_auc = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(label, var_score)
        var_roc_auc = auc(fpr, tpr)

        df = pd.DataFrame(ftr_similarity)
        df.to_csv(test_dir + 'pred/ftr_similarity_' + str(subdim) + '.csv')
        df = pd.DataFrame(mu_similarity)
        df.to_csv(test_dir + 'pred/mu_similarity_' + str(subdim) + '.csv')
        df = pd.DataFrame(var_similarity)
        df.to_csv(test_dir + 'pred/var_similarity_' + str(subdim) + '.csv')
        df = pd.DataFrame(list(zip(y, label, ftr_score)))
        df.to_csv(test_dir + "pred/ftr_score_" + str(subdim) + ".csv")
        df = pd.DataFrame(list(zip(y, label, mu_score)))
        df.to_csv(test_dir + "pred/mu_score_" + str(subdim) + ".csv")
        df = pd.DataFrame(list(zip(y, label, var_score)))
        df.to_csv(test_dir + "pred/var_score_" + str(subdim) + ".csv")

        subdim_result[subdim, 0] = subdim
        subdim_result[subdim, 1] = accuracy_score(y, ftr_prediction)
        subdim_result[subdim, 2] = accuracy_score(y, mu_prediction)
        subdim_result[subdim, 3] = accuracy_score(y, var_prediction)
        subdim_result[subdim, 4] = ftr_roc_auc
        subdim_result[subdim, 5] = mu_roc_auc
        subdim_result[subdim, 6] = var_roc_auc

        print('Classification:')
        print('混合行列\n', confusion_matrix(y, ftr_prediction),'\n', confusion_matrix(y, mu_prediction),'\n',  confusion_matrix(y, var_prediction))
        print('accuracy = ', accuracy_score(y, ftr_prediction), accuracy_score(y, mu_prediction), accuracy_score(y, var_prediction))
        print('pred (ftr, mu, var) = ', ftr_prediction, mu_prediction, var_prediction)
        
        print('AnomalyDetection: ')
        print('roc = ', ftr_roc_auc, mu_roc_auc, var_roc_auc)

        print('subdim = ', subdim)


    df = pd.DataFrame(subdim_result)
    df.to_csv(test_dir + 'pred/000_Subdim_Result.csv')



def kernel_subspace_test():
    # 辞書空間にするデータを取ってくる　データのならびはftr, mu, varの全データ×512がクラスラベル0~6順に並ぶ
    ftr, mu, var, clsnm, num, _ = get_features_and_sort(dictionary_dir)

    print('Anomaly_Class:', Anomaly_object)

    #TrainData
    print('TrainData: clsnm', clsnm, 'num', num, '\n', [dic_names[str(clsnm[k])] for k in range(CLASS_NUM)])

    # TrainDataからAnomalyClassを排除
    ftr, clsnm, num = exclude_anomaly_obj(ftr, clsnm, num, Anomaly_object)

    subspace_class_num = CLASS_NUM - 1
    #print('TrainData (without AnomalyClass): clsnm', clsnm, 'num', num, '\n', [dic_names[str(clsnm[k])] for k in range(subspace_class_num)])
    print('\n<TrainData (without AnomalyClass):>')
    for i in range(subspace_class_num):
        print(clsnm[i], num[i], ':', num[i+1], dic_names[str(clsnm[i])])

    # TrainDataをlistにまとめる
    X_train_ftr = []
    X_train_mu = []
    X_train_var = []
    for i in range(subspace_class_num):
        X_train_ftr.append((ftr[num[i]:num[i+1], :].T))
        X_train_mu.append((mu[num[i]:num[i+1], :].T))
        X_train_var.append((var[num[i]:num[i+1], :].T))
    
    labels = list(range(subspace_class_num)) # [0, ~ ,6]
        
    # TestDataを取ってきて、AnomalyClassのlabelを1にセット
    test_ftr, test_mu, test_var, test_clsnm, test_num, y_test = get_features_and_sort(test_dir)
    anomaly_labels = set_anomaly_labels(test_ftr, test_clsnm, test_num, Anomaly_object)

    print('\n<Testdata>:')
    for i in range(CLASS_NUM):
        print(test_clsnm[i], test_num[i], ':', test_num[i+1], dic_names_test[str(test_clsnm[i])])
    
    print('\n<Traindata anomaly_labels>:')
    for i, k, in enumerate(dic_names_test):
        print(k, dic_names_test[str(test_clsnm[i])],'\t', anomaly_labels[test_num[i]])

    # kernel PCA
    ksm.kernel_subspace_anomaly_detection(X_train_ftr, labels, test_ftr.T, y_test, anomaly_labels, knl_dir)
    #ksm.calc_kernel_subspace_bases(X_train_ftr, labels, test_ftr.T, y_test, anomaly_labels, knl_dir)

    print('Done')
    

kernel_subspace_test()


# for i in range (CLASS_NUM):
#     dt = ftr[num[i]:num[i+1], :].T
#     test_ftr_bases[i] = calc_subspace(ftr[num[i]:num[i+1], :].T, subdim)
#     test_m_bases[i] = calc_subspace(mu[num[i]:num[i+1], :].T, subdim)
#     test_v_bases[i] = calc_subspace(var[num[i]:num[i+1], :].T, subdim)



# # 類似度計算
# ftr_sim = np.zeros((CLASS_NUM, CLASS_NUM))
# m_sim = np.zeros((CLASS_NUM, CLASS_NUM))
# v_sim = np.zeros((CLASS_NUM, CLASS_NUM))
# for i in range (CLASS_NUM):
#     for j in range (CLASS_NUM):
#         ftr_sim[i,j] = calc_similarity(test_ftr_bases[i], dic_ftr_bases[j])
#         m_sim[i,j] = calc_similarity(test_m_bases[i], dic_m_bases[j])
#         v_sim[i,j] = calc_similarity(test_v_bases[i], dic_v_bases[j])


# for i in range(CLASS_NUM):
#     print('ftr_sim', ftr_sim[i,:], np.argmax(ftr_sim[i,:]))
#     print('m_sim', m_sim[i,:], np.argmax(m_sim[i,:]))
#     print('v_sim', v_sim[i,:], np.argmax(v_sim[i,:]))