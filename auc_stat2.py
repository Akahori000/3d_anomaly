## kernel結果を可視化など

import numpy as np
import pandas as pd
import os
import glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve
import sys

CLASS_NUM = 7
testn = {'lamp':'928', 'chair':'756', 'table':'', 'car':'280', 'sofa':'508', 'rifle':'404', 'airplane':'576'} # test のとき
valn = {'lamp':'464', 'chair':'378', 'table':'', 'car':'140', 'sofa':'254', 'rifle':'204', 'airplane':'288'} # val のとき

names = ['lamp', 'chair', 'table', 'car', 'sofa', 'rifle', 'airplane']
epoclist = [150, 200, 250, 299]



# A, B がそれぞれ基底行列 　(A.T@B) の特異値分解の特異値⇒正準角のcos 　
# dirを読み込んでfeature, mu, varをクラス毎(numに情報あり)に並び替えて返す関数
def get_features_and_sort(dir):
    #features = pd.read_csv(dir + 'feature.csv')
    nms = pd.read_csv(dir + 'name.csv')
    mu = pd.read_csv(dir + 'mu.csv')
    #var = pd.read_csv(dir + 'var.csv')

    #features = features.iloc[:,1:].values
    nms = nms.iloc[:, 1:].values.ravel()
    mu = mu.iloc[:, 1:].values
    #var = var.iloc[:, 1:].values

    clsnm = np.zeros(CLASS_NUM) # 各クラスのデータ数
    #ftr = np.zeros((features.shape)) # クラスごと並び替え
    m = np.zeros((mu.shape))
    #v = np.zeros((var.shape))
    y = np.zeros(mu.shape[0])
    label = np.zeros(mu.shape[0]) #normal = 0, abnormal(anomaly) = 1 

    for cnt in range(CLASS_NUM):
        idx = [i for i, x in enumerate(nms.tolist()) if x == cnt] # 特定のクラスのindexを抽出
        #ftr[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = features[idx, :]
        m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
        #v[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = var[idx, :]
        y[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx)))] = cnt
        clsnm[cnt] = int(len(idx))

    clsnm = np.array(clsnm, dtype=int)
    num = np.array([0, clsnm[0], np.sum(clsnm[:2]), np.sum(clsnm[:3]), np.sum(clsnm[:4]), np.sum(clsnm[:5]), np.sum(clsnm[:6]), np.sum(clsnm)])
    label[num[6]: num[7]] = 1

    return m, clsnm, num, y   # feature, mu, var, y 


def set_anomaly_labels(ftr, clsnm, num, anomaly_obj):

    anomaly_labels = np.zeros(len(ftr))

    anomaly_num = names.index(anomaly_obj)
    anomaly_labels[num[anomaly_num]: num[anomaly_num+1]] = 1 
    return anomaly_labels


def draw_auc_roc_thresh(anomaly_labels, anomaly_scores, subdim, gamma, savepath):

    # AUCの計算
    fpr, tpr, thresh = roc_curve(anomaly_labels, anomaly_scores)
    ftr_roc_auc = auc(fpr, tpr)
    #print(ftr_roc_auc)
    
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(savepath + '001_linearSM_AUCmax_ROC_subdim' + str(subdim) + '_gamma' + str(gamma) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()


    plt.plot(np.array(range(len(thresh))), thresh, marker='o')
    plt.ylabel('Threshold')
    plt.title('Threshold')
    plt.grid()
    plt.savefig(savepath + '002_linearSM_AUCmax_thresh_subdim' + str(subdim) + '_gamma' + str(gamma) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()

    plt.plot(np.array(range(len(anomaly_scores))), anomaly_scores, marker='o')
    plt.ylabel('anomaly_scores')
    plt.title('anomaly_scores')
    plt.xlim(0, len(anomaly_scores))
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(savepath + '003_linearSM_AUCmax_score_subdim' + str(subdim) + '_gamma' + str(gamma) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()


def draw_auc_roc_thresh_test(anomaly_labels, anomaly_scores, subdim, gamma, savepath):

    # AUCの計算
    fpr, tpr, thresh = roc_curve(anomaly_labels, anomaly_scores)
    ftr_roc_auc = auc(fpr, tpr)
    print('test', ftr_roc_auc)
    
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(savepath + '000_linearSM_AUCmax_ROC_subdim' + str(subdim) + '_gamma' + str(gamma) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()


    plt.plot(np.array(range(len(thresh))), thresh, marker='o')
    plt.ylabel('Threshold')
    plt.title('Threshold')
    plt.grid()
    plt.savefig(savepath + '000_linearSM_AUCmax_thresh_subdim' + str(subdim) + '_gamma' + str(gamma) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()

    plt.plot(np.array(range(len(anomaly_scores))), anomaly_scores, marker='o')
    plt.ylabel('anomaly_scores')
    plt.title('anomaly_scores')
    plt.xlim(0, len(anomaly_scores))
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(savepath + '000_linearSM_AUCmax_score_subdim' + str(subdim) + '_gamma' + str(gamma) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()
    
    return ftr_roc_auc


def draw_auc_roc_thresh_test_linear(anomaly_labels, anomaly_scores, subdim, savepath):

    # AUCの計算
    fpr, tpr, thresh = roc_curve(anomaly_labels, anomaly_scores)
    ftr_roc_auc = auc(fpr, tpr)
    print('test', ftr_roc_auc)
    
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(savepath + '000_linearSM_AUCmax_ROC_subdim' + str(subdim) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()


    plt.plot(np.array(range(len(thresh))), thresh, marker='o')
    plt.ylabel('Threshold')
    plt.title('Threshold')
    plt.grid()
    plt.savefig(savepath + '000_linearSM_AUCmax_thresh_subdim' + str(subdim) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()

    plt.plot(np.array(range(len(anomaly_scores))), anomaly_scores, marker='o')
    plt.ylabel('anomaly_scores')
    plt.title('anomaly_scores')
    plt.xlim(0, len(anomaly_scores))
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(savepath + '000_linearSM_AUCmax_score_subdim' + str(subdim) + '_auc' + str(round(ftr_roc_auc, 4)) + '.png')
    plt.clf()
    
    return ftr_roc_auc


def kernelSM_data_analyze():
    arr = np.zeros((4, CLASS_NUM))
    arr_test = np.zeros((4, CLASS_NUM))

    for i in range(CLASS_NUM):
        #anomaly_class = names[i]
        anomaly_class = 'car'
        if anomaly_class != 'table':    #tableはまだ学習できてないので
            for j in range(4):
                epocn = epoclist[j]
                epoc = str(epocn)    # 150, 200, 250, 299


                path = './data/calculated_features/modelAE_' + anomaly_class + '/both_features/c_epoc_' + epoc + '_data' + valn[anomaly_class] + '/'
                print(path)
                kernel_path = path + 'kernel_pred2/'
                auc_path = kernel_path + 'auc.csv'


                df = pd.read_csv(auc_path)
                aucs = df.iloc[:, 1:-1].values.tolist()
                # 最大値とそのindex
                max_auc = np.max(aucs)
                idx = np.argwhere(aucs == max_auc).reshape(2)
                print(max_auc, idx)
                arr[j, i] = max_auc

                # subdimとgamma
                subdim = idx[1] + 1
                gamma = idx[0] * 10 + 1
                print('subdim:', subdim, 'gamma:', gamma)
                ascr_path = kernel_path + 'auc_subdim' + str(subdim) + '_gamma' + str(gamma) + '.csv'

                ascr = pd.read_csv(ascr_path)
                anomaly_score = ascr.iloc[:, -1].values
                #print(anomaly_score)

                test_ftr, test_clsnm, test_num, y = get_features_and_sort(path)
                anomaly_labels = set_anomaly_labels(test_ftr, test_clsnm, test_num, anomaly_class)

                draw_auc_roc_thresh(anomaly_labels, anomaly_score, subdim, gamma, kernel_path)

                # #valから求めたsubdimとgammaからaucなど求める
                # path = './data/calculated_features/modelAE_' + anomaly_class + '/both_features/c_epoc_' + epoc + '_data' + testn[anomaly_class] + '/'
                # kernel_path = path + 'kernel_pred2/'
                # ascr_path = kernel_path + 'auc_subdim' + str(subdim) + '_gamma' + str(gamma) + '.csv'
                # print(ascr_path)
                # ascr = pd.read_csv(ascr_path)
                # anomaly_score = ascr.iloc[:, -1].values

                # test_ftr, test_clsnm, test_num, y = get_features_and_sort(path)
                # anomaly_labels = set_anomaly_labels(test_ftr, test_clsnm, test_num, anomaly_class)

                # arr_test[j, i] = draw_auc_roc_thresh_test(anomaly_labels, anomaly_score, subdim, gamma, kernel_path)


    df = pd.DataFrame(arr)
    df.to_csv('./data/calculated_features/00_result/AE_kernel_result_val.csv')

    # df = pd.DataFrame(arr_test)
    # df.to_csv('./data/calculated_features/00_result/AE_kernel_result_test_using_val.csv')


def LinearSM_data_analyze():
    arr = np.zeros((4, CLASS_NUM))
    arr_test = np.zeros((4, CLASS_NUM))

    for i in range(CLASS_NUM):
        anomaly_class = names[i]
        anomaly_class = 'car'
        if anomaly_class != 'table':    #tableはまだ学習できてないので
            for j in range(4):
                epocn = epoclist[j]
                epoc = str(epocn)    # 150, 200, 250, 299

                # validataionデータでAUC最大のparam取得
                path = './data/calculated_features/modelAE_' + anomaly_class + '/both_features/c_epoc_' + epoc + '_data' + valn[anomaly_class] + '/'
                print(path)
                linear_path = path + 'pred_NA_halfhalf2/'
                auc_path = linear_path + '000_Subdim_Result.csv'
                df = pd.read_csv(auc_path)
                aucs = df.iloc[1:,-1:].values.tolist()
                max_auc = np.max(aucs)
                arr[j, i] = max_auc
                subdim = np.argwhere(aucs == max_auc)[0,0]
                subdim += 1
                print('val max_subdim:', subdim, max_auc)


                #testデータで最大だったときのAUC取得
                path = './data/calculated_features/modelAE_' + anomaly_class + '/both_features/c_epoc_' + epoc + '_data' + testn[anomaly_class] + '/'
                print(path)
                linear_path = path + 'pred_NA_halfhalf2/'
                auc_path = linear_path + '000_Subdim_Result.csv'
                df = pd.read_csv(auc_path)
                aucs = df.iloc[1:,-1:].values.tolist()
                max_auc_in_test = aucs[subdim - 1]
                print('max_auc_in_test', max_auc_in_test)

                ascr_path = linear_path + 'ftr_score_' + str(subdim) + '.csv'
                ascr = pd.read_csv(ascr_path)
                anomaly_score = ascr.iloc[:, -1].values

                test_ftr, test_clsnm, test_num, y = get_features_and_sort(path)
                anomaly_labels = set_anomaly_labels(test_ftr, test_clsnm, test_num, anomaly_class)

                arr_test[j, i] = draw_auc_roc_thresh_test_linear(anomaly_labels, anomaly_score, subdim, linear_path)


    df = pd.DataFrame(arr)
    df.to_csv('./data/calculated_features/00_result/AE_linear_result_val.csv')

    df = pd.DataFrame(arr_test)
    df.to_csv('./data/calculated_features/00_result/AE_linear_result_test_using_val.csv')

#kernelSM_data_analyze()
LinearSM_data_analyze()