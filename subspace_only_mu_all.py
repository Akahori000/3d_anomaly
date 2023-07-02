import numpy as np
import pandas as pd
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
import sys

from ksm_example import ksm_exp as ksm


CLASS_NUM = 14
args = sys.argv
Anomaly_object = args[1]
#Anomaly_object = 'airplane' # ←ここでobject指定すること!
test_folder = args[2]
#test_folder = 'c_epoc_299_data576' #←例

test_way = args[3]

dic_folder = args[4]
#dic_folder = 'c_epoc_299_data2521' #←例


dictionary_dir =  './data/shapenet/objset3/calculated_features_all/modelAE_' + Anomaly_object + '/both_features/' + dic_folder + '/'
test_dir = './data/shapenet/objset3/calculated_features_all/modelAE_' + Anomaly_object + '/both_features/' + test_folder + '/'
knl_dir = test_dir + 'kernel_pred_all/'
lnr_dir = test_dir + 'pred/'
lnr_dir_hlf = test_dir + 'pred_NA_halfhalf_all/' # これが2の方が|| A@A.T@x|| で求めた方

dic_names = {'1622': 'lamp', '1323': 'chair', '1078':'table', '490':'car', '890':'sofa', '709': 'rifle', '1007': 'airplane', '317': 'bookshelf', '322': 'laptop', '297':'knife', '272':'train', '236':'motorbike', '557': 'guitar', '520': 'faucet'}
if test_way == 'test':
    dic_names_test = {'464': 'lamp', '378': 'chair', '308':'table', '140':'car', '254':'sofa', '202': 'rifle', '288': 'airplane', '90': 'bookshelf', '92': 'laptop', '85':'knife', '78':'train', '67':'motorbike', '160': 'guitar', '149': 'faucet'}
else:
    dic_names_test = {'232': 'lamp', '189': 'chair', '154':'table', '70':'car', '127':'sofa', '102': 'rifle', '144': 'airplane', '45': 'bookshelf', '46': 'laptop', '42':'knife', '39':'train', '34':'motorbike', '80': 'guitar', '75': 'faucet'} # val用
names = ['lamp', 'chair', 'table', 'car', 'sofa', 'rifle', 'airplane', 'bookshelf', 'laptop', 'knife', 'train', 'motorbike', 'guitar', 'faucet']

# dic_names = {'317': 'bookshelf', '322': 'laptop', '297':'knife', '272':'train', '236':'motorbike', '557': 'guitar', '520': 'faucet'}
# if test_way == 'test':
#     dic_names_test = {'90': 'bookshelf', '92': 'laptop', '85':'knife', '78':'train', '67':'motorbike', '160': 'guitar', '149': 'faucet'}
# else:
#     dic_names_test = {'45': 'bookshelf', '46': 'laptop', '42':'knife', '39':'train', '34':'motorbike', '80': 'guitar', '75': 'faucet'} # val用
# names = ['bookshelf', 'laptop', 'knife', 'train', 'motorbike', 'guitar', 'faucet']


# dic_names = {'1622': 'lamp', '1323': 'chair', '1078':'table', '490':'car', '890':'sofa', '709': 'rifle', '1007': 'airplane'}
# names = ['lamp', 'chair', 'table', 'car', 'sofa', 'rifle', 'airplane']

# if test_way == 'test':
#     dic_names_test = {'464': 'lamp', '378': 'chair', '308':'table', '140':'car', '254':'sofa', '202': 'rifle', '288': 'airplane'}
# elif test_way == 'val':
#     dic_names_test = {'232': 'lamp', '189': 'chair', '154':'table', '70':'car', '127':'sofa', '102': 'rifle', '144': 'airplane'} # val用

test_data_sum = 9640 #c_epoc_100_data2755 みたいな全テストデータの数  objset1:2755, objset2:721, objset3: 9640

if not os.path.exists(lnr_dir):
    os.makedirs(lnr_dir)
if not os.path.exists(knl_dir):
    os.makedirs(knl_dir)
if not os.path.exists(lnr_dir_hlf):
    os.makedirs(lnr_dir_hlf)

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

# 金井さんおすすめ基底とベクトルの射影長求める手法
def calc_projection_len(input_vector, basis_vector):
    norm = np.linalg.norm(basis_vector @ basis_vector.T @ input_vector, axis=0)
    return(norm)

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

    for cnt in range(CLASS_NUM):
        idx = [i for i, x in enumerate(names.tolist()) if x == cnt] # 特定のクラスのindexを抽出
        ftr[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = features[idx, :]
        m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
        v[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = var[idx, :]
        y[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx)))] = cnt
        clsnm[cnt] = int(len(idx))

    clsnm = np.array(clsnm, dtype=int)
    num = np.zeros(CLASS_NUM + 1)
    for i in range(CLASS_NUM + 1):
        num[i] = int(np.sum(clsnm[:i]))
    num = np.array(num, dtype=int)
    
    return ftr, m, v, clsnm, num, y   # feature, mu, var, y 


def get_features_and_sort_onlymu(dir):
    names = pd.read_csv(dir + 'name.csv')
    mu = pd.read_csv(dir + 'mu.csv')
    
    names = names.iloc[:, 1:].values.ravel()
    mu = mu.iloc[:, 1:].values
    
    clsnm = np.zeros(CLASS_NUM) # 各クラスのデータ数
    m = np.zeros((mu.shape))
    y = np.zeros(mu.shape[0])

    for cnt in range(CLASS_NUM):
        idx = [i for i, x in enumerate(names.tolist()) if x == cnt] # 特定のクラスのindexを抽出
        m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
        y[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx)))] = cnt
        clsnm[cnt] = int(len(idx))

    clsnm = np.array(clsnm, dtype=int)
    num = np.zeros(CLASS_NUM + 1)
    for i in range(CLASS_NUM + 1):
        num[i] = int(np.sum(clsnm[:i]))
    num = np.array(num, dtype=int)

    return m, clsnm, num, y   # feature, mu, var, y 

def set_anomaly_labels(ftr, clsnm, num, anomaly_obj):

    anomaly_labels = np.zeros(len(ftr))

    anomaly_num = [k for k, v in dic_names_test.items() if v == anomaly_obj]
    anomaly_num = int(anomaly_num[0])
    
    for i in range(len(clsnm)):
        if clsnm[i] == anomaly_num:
            anomaly_labels[num[i]: num[i+1]] = 1 

    return anomaly_labels

def set_anomaly_labels_simple(ftr, clsnm, num, anomaly_obj):

    anomaly_labels = np.zeros(len(ftr))

    anomaly_num = names.index(anomaly_obj)
    anomaly_labels[num[anomaly_num]: num[anomaly_num+1]] = 1 
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
    
    num = np.zeros(CLASS_NUM)
    for i in range(CLASS_NUM):
        num[i] = int(np.sum(clsnm[:i]))
    num = np.array(num, dtype=int)

    #print('exclude anomaly_obj (train)', anomaly_obj, anomaly_num, len(ftr)-len(ftre))

    return ftre, clsnm, num


# 部分空間法で異常検知
def linear_subspace_anomaly_detection():
    # テスト開始
    sub_dim = 80
    train_ftr, clsnm, num, _ = get_features_and_sort_onlymu(dictionary_dir)

    print('Anomaly_Class:', Anomaly_object)

    #TrainData
    print('TrainData: clsnm', clsnm, 'num', num, '\n', [dic_names[str(clsnm[k])] for k in range(CLASS_NUM)])

    # TrainDataからAnomalyClassを排除
    ftr, clsnm, num = exclude_anomaly_obj(train_ftr, clsnm, num, Anomaly_object)
    subspace_class_num = CLASS_NUM - 1

    print('\n<TrainData (without AnomalyClass):>')
    for i in range(subspace_class_num):
        print(clsnm[i], num[i], ':', num[i+1], dic_names[str(clsnm[i])])

    # テストデータ
    test_ftr, test_clsnm, test_num, y = get_features_and_sort_onlymu(test_dir)
    anomaly_labels = set_anomaly_labels(test_ftr, test_clsnm, test_num, Anomaly_object)


    if sum(test_clsnm) < test_data_sum:
        print('\n<Test_datanum & Test_datanum & Anomaly_Labels>:')
        for i in range(CLASS_NUM):
            print(names[i], test_clsnm[i],'\t',int(anomaly_labels[test_num[i]]))

        lnr_dir = lnr_dir_hlf
        if not os.path.exists(lnr_dir):
            os.makedirs(lnr_dir)

    else:
        print('\n<Testdata>:')
        for i in range(CLASS_NUM):
            print(test_clsnm[i], test_num[i], ':', test_num[i+1], dic_names_test[str(test_clsnm[i])])
        
        print('\n<Traindata anomaly_labels>:')
        for i, k, in enumerate(dic_names_test):
            print(k, dic_names_test[str(test_clsnm[i])],'\t', anomaly_labels[test_num[i]])


    subdim_result = np.zeros((sub_dim + 1, 2))  #subdim, auc

    # 部分空間の次元を変化させてテスト
    for subdim in (n+1 for n in range(sub_dim)):
        # 辞書データ
        #dic_ftr_bases = np.zeros((subspace_class_num, ftr.shape[1], subdim))
        #for i in range(subspace_class_num):
        #    dic_ftr_bases[i] = calc_subspace(ftr[num[i]:num[i+1], :].T, subdim)

        # 辞書データ (subspace1個)
        dic_ftr_bases = np.zeros((ftr.shape[1], subdim))
        dic_ftr_bases = calc_subspace(ftr.T, subdim)

        # 各テストデータで行う
        testdt_num = test_ftr.shape[0]
        ftr_similarity = np.zeros(testdt_num)

        ftr_score = np.zeros(testdt_num)

        for i in range(testdt_num):  # 全データ
            ftr_similarity[i] = calc_projection_len((test_ftr[i, :].T).reshape(test_ftr.shape[1], 1), dic_ftr_bases)

            # for j in range(subspace_class_num):  # 各クラス
            #     ftr_similarity[i, j] = calc_projection_len((test_ftr[i, :].T).reshape(test_ftr.shape[1], 1), dic_ftr_bases[j])

            # 異常度の計算　 1- (テストデータと辞書空間のcos類似度の最大になるクラスのcos類似度)
            #ftr_score[i] = 1 - max(ftr_similarity[i,:])

            ftr_score[i] = 1 - (ftr_similarity[i])


        # anomaly_scoreを0~1に正規化
        _min = np.array(min(ftr_score))
        _max = np.array(max(ftr_score))

        re_scaled = (ftr_score - _min) / (_max - _min)
        re_scaled = np.array(re_scaled, dtype=float)

        # AUCの計算
        fpr, tpr, thresh = roc_curve(anomaly_labels, re_scaled)
        ftr_roc_auc = auc(fpr, tpr)

        df = pd.DataFrame(ftr_similarity)
        df.to_csv(lnr_dir + 'ftr_similarity_' + str(subdim) + '.csv')
        df = pd.DataFrame(list(zip(y, anomaly_labels, re_scaled)))
        df.to_csv(lnr_dir + '/ftr_score_' + str(subdim) + ".csv")
        df = pd.DataFrame(thresh)
        df.to_csv(lnr_dir + '/ftr_thresh_' + str(subdim) + ".csv")

        subdim_result[subdim, 0] = subdim
        subdim_result[subdim, 1] = ftr_roc_auc

        print('subdim = ', subdim, ', roc = ', ftr_roc_auc)

    df = pd.DataFrame(subdim_result)
    df.to_csv(lnr_dir + '000_Subdim_Result.csv')
    return(lnr_dir)


# 最大のAUCを持つ条件のAnomalyScoreのcsvからROCカーブ、Thresh、AnomalyScoreのグラフを描画
def get_the_max_auc_roc(file_dir):
    print(Anomaly_object)
    dt = pd.read_csv(file_dir + '000_Subdim_Result.csv')
    temp = dt.values[1:,-1:]
    subdim = ([i for i, x in enumerate(temp) if x == max(temp)])[0] + 1 # AUCが最大のときの部分空間の次元
    sim = pd.read_csv(file_dir + '/ftr_score_' + str(subdim) + ".csv")
    anomaly_labels = sim.iloc[:,-2].to_numpy()
    anomaly_scores = sim.iloc[:,-1].to_numpy()

    # AUCの計算
    _min = np.array(min(anomaly_scores))
    _max = np.array(max(anomaly_scores))

    re_scaled = (anomaly_scores - _min) / (_max - _min)
    re_scaled = np.array(re_scaled, dtype=float)


    # AUCの計算
    fpr, tpr, thresh = roc_curve(anomaly_labels, re_scaled)
    ftr_roc_auc = auc(fpr, tpr)
    print(ftr_roc_auc)
    
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(file_dir + '001_linearSM_AUCmax_ROC_subdim' + str(subdim) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()


    plt.plot(np.array(range(len(thresh))), thresh, marker='o')
    plt.ylabel('Threshold')
    plt.title('Threshold')
    plt.grid()
    plt.savefig(file_dir + '002_linearSM_AUCmax_thresh_subdim' + str(subdim) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()

    plt.plot(np.array(range(len(re_scaled))), re_scaled, marker='o')
    plt.ylabel('anomaly_scores')
    plt.title('anomaly_scores')
    plt.xlim(0, len(re_scaled))
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(file_dir + '003_linearSM_AUCmax_thresh_scores' + str(subdim) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()

    return(dt)


def kernel_subspace_test():
    # 辞書空間にするデータを取ってくる　データのならびはftr, mu, varの全データ×512がクラスラベル0~6順に並ぶ
    ftr, clsnm, num, _ = get_features_and_sort_onlymu(dictionary_dir)

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
    X_train_mu = []
    # for i in range(subspace_class_num):
    #     X_train_mu.append((ftr[num[i]:num[i+1], :].T))
    X_train_mu = ftr.T
    
    labels = list(range(subspace_class_num)) # [0, ~ ,6]
        
    # TestDataを取ってきて、AnomalyClassのlabelを1にセット
    test_mu, test_clsnm, test_num, y_test = get_features_and_sort_onlymu(test_dir)
    anomaly_labels = set_anomaly_labels(test_mu, test_clsnm, test_num, Anomaly_object)

    if sum(test_clsnm) < test_data_sum:
        print('\n<Test_datanum & Test_datanum & Anomaly_Labels>:')
        for i in range(CLASS_NUM):
            print(names[i], test_clsnm[i],'\t',int(anomaly_labels[test_num[i]]))

        lnr_dir = lnr_dir_hlf
        if not os.path.exists(lnr_dir):
            os.makedirs(lnr_dir)
    else:
        print('\n<Testdata>:')
        for i in range(CLASS_NUM):
            print(test_clsnm[i], test_num[i], ':', test_num[i+1], dic_names_test[str(test_clsnm[i])])
        
        print('\n<Traindata anomaly_labels>:')
        for i, k, in enumerate(dic_names_test):
            print(k, dic_names_test[str(test_clsnm[i])],'\t', anomaly_labels[test_num[i]])

    # kernel PCA
    ksm.save_already_calculated_ones(knl_dir)
    ksm.kernel_subspace_anomaly_detection_all(X_train_mu, labels, test_mu.T, y_test, anomaly_labels, knl_dir)
    #ksm.calc_kernel_subspace_bases(X_train_mu, labels, test_mu.T, y_test, anomaly_labels, knl_dir)

    print('Done')
    

if Anomaly_object in names:

    # file_dir = linear_subspace_anomaly_detection()
    # print(file_dir)
    # get_the_max_auc_roc(file_dir) # このfile_dirは'000_Subdim_Result.csv'のある場所　AUCの測り方によってちがう
    
    kernel_subspace_test()
else:
    print('command line input error')
