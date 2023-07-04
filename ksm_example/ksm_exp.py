# %%
import time
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
#import toolbox as tb

from .toolbox import kernel as kn
from .toolbox import functional as fn

POLLEN_ROOT = "/mnt/sda1/datasets/pollen"
FEATURES_ROOT = f"{POLLEN_ROOT}/features"

RANDOM_SEED = 1337
val_size = 0.3

######
# load and prepare data
def load_and_prepare_data():
    X_full = loadmat("/mnt/sda1/datasets/pollen/features/pollen13k_subset_wrn101_fine.mat")["data"]

    X_train = []
    X_test = []

    for _X in X_full[0]:
        _X_train, _X_val = train_test_split(
            _X,
            random_state=RANDOM_SEED,
            test_size=val_size,
        )
        X_train.append(_X_train)
        X_test.append(_X_val)

    y_test = X_test[0].shape[0] * [0]
    y_test.extend(X_test[1].shape[0] * [1])
    y_test.extend(X_test[2].shape[0] * [2])
    y_test.extend(X_test[3].shape[0] * [3])

    X_test = np.concatenate(X_test)

    X_train = [_X.T for _X in X_train]
    X_test = X_test.T

    labels = [0, 1, 2, 3]

    return X_train, labels, X_test, y_test

# %%

# 正常クラスのみで学習
def calc_anomaly_score(kernel_similarities):
    data_num = len(kernel_similarities)
    class_num = len(kernel_similarities[0])
    anomaly_score = np.zeros(data_num)
    for i in range(data_num):
        sims = kernel_similarities[i, :]
        anomaly_score[i] = 1 - np.max(sims)

    return  anomaly_score


def save_already_calculated_ones(knl_dir):

    search_subdim = range(1, 150, 1)
    search_gammas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.2236, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                     11, 12, 14, 16, 18, 20, 21, 25, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291]
    
    aucs = np.zeros((len(search_gammas), len(search_subdim)))
    for cnti,i in enumerate(search_subdim):
        for cntj, j in enumerate(search_gammas):
            if os.path.exists(knl_dir + 'auc_subdim' + str(i) + '_gamma' + str(j) +  '.csv'):
                df = pd.read_csv(knl_dir + 'auc_subdim' + str(i) + '_gamma' + str(j) +  '.csv')
                scrs = df.values[:,2]
                lbls = df.values[:,1]
                fpr, tpr, th = roc_curve(lbls, scrs)
                aucs[cntj, cnti] = auc(fpr, tpr)
    df1 = pd.DataFrame(aucs)
    df1 = df1.set_axis(search_subdim, axis=1)
    df1 = df1.set_axis(search_gammas, axis=0)
    df1.to_csv(knl_dir + '0_current_aucs.csv')



def kernel_subspace_anomaly_detection_all(X_train, labels, X_test, y_test, anomaly_labels, knl_dir):
    #n_subdims = range(1, 150, 1)
    #gammas = range(1, 300, 10)    

    # n_subdims = range(1, 150, 1)
    n_subdims = range(1, 90, 1)
    temp = [90, 100, 130, 150]
    n_subdims = list(n_subdims)
    n_subdims.extend(temp)

    gammas = [0.01, 0.0236, 0.05, 0.1, 0.2, 0.2236, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1,2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,25,31,41,51,61,71,81,91,101,131,151,171,201,251,291]
    # #gammas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.2236, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    #           11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291]


    df = pd.DataFrame(columns=["n_subdims", "gamma", "acc", "weighted_f1", "macro_f1", "fit_time", "predict_time"])
    ftr_roc_auc = np.zeros((len(gammas), len(n_subdims)))
    thresh = np.zeros((len(gammas), len(n_subdims)))

    for i, n_subdim in enumerate(n_subdims):
        for j, gamma in enumerate(gammas):

            if ((os.path.exists(knl_dir + 'auc_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv') == False) and
                (os.path.exists(knl_dir + 'auc_subdim' + str(n_subdim) + '_gamma' + str(gamma) + ".0" +  '.csv') == False)) :
                # kernel_bases = [
                #    kn.kernel_subspace_bases(X_class, n_subdim, gamma) for X_class in X_train
                # ]
                print("X_train_len", X_train.shape, "gamma", gamma, "n_subdim", n_subdim)
                kernel_bases = [
                kn.kernel_subspace_bases(X_train, n_subdim, gamma)
                ]
                print('kernel_basis', len(kernel_bases))
                kernel_bases = np.array(kernel_bases)
                #print(f"Kernel fit time: {fit_time}")

                # kernel_similarities = [
                #     kn.kernel_similarity(_kernel_base, _X, X_test)
                #     for _kernel_base, _X in zip(kernel_bases, X_train)
                # ]

                kernel_similarities = [
                    kn.kernel_similarity(kernel_bases.reshape(kernel_bases.shape[1],kernel_bases.shape[2]), X_train, X_test)
                ]
                kernel_similarities = np.vstack(kernel_similarities).T
                # print('kernelsimilarities', len(kernel_similarities))

                # 正常クラスのみでanomalyscoreを出す
                #pred = calc_anomaly_score(kernel_similarities[:, :(kernel_similarities.shape[1] - 1)])
                pred = np.ones(len(kernel_similarities)) - kernel_similarities.reshape(len(kernel_similarities))
                
                # AUCの計算
                _min = np.array(min(pred))
                _max = np.array(max(pred))

                re_scaled = (pred - _min) / (_max - _min)
                re_scaled = np.array(re_scaled, dtype=float)
                fpr, tpr, th = roc_curve(anomaly_labels, re_scaled)
                ftr_roc_auc[j, i] = auc(fpr, tpr)
                print(f"Subspace dimensions: {n_subdim}", f"Gamma: {gamma}", 'AUC', ftr_roc_auc[j, i])

                # plt.plot(fpr, tpr, marker='o')
                # plt.xlabel('FPR: False positive rate')
                # plt.ylabel('TPR: True positive rate')
                # plt.grid()
                # plt.savefig(knl_dir + 'ROCCurve_subdim' + str(n_subdim) + '_gamma' + str(gamma) + '.png')
                # plt.clf()

                
                df3 = pd.DataFrame(np.vstack([anomaly_labels, re_scaled]).T)
                df3.to_csv(knl_dir + 'auc_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv')
                df3 = pd.DataFrame(th)
                df3.to_csv(knl_dir + 'thresh_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv')

    df1 = pd.DataFrame(ftr_roc_auc)
    df1.to_csv(knl_dir + '0_auc_gamma_all.csv')

    #df1 = pd.DataFrame(ftr_roc_auc)
    #df1.to_csv(knl_dir + 'auc.csv')



def kernel_subspace_anomaly_detection(X_train, labels, X_test, y_test, anomaly_labels, knl_dir):
    #n_subdims = range(1, 150, 1)
    #gammas = range(1, 300, 10)

    n_subdims = range(1, 150, 1)
    gammas = [0.01, 0.05, 0.1, 0.2236, 0.5, 1, 5, 10, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291]

    df = pd.DataFrame(columns=["n_subdims", "gamma", "acc", "weighted_f1", "macro_f1", "fit_time", "predict_time"])
    ftr_roc_auc = np.zeros((len(gammas), 150))
    thresh = np.zeros((len(gammas), 150))

    for i, n_subdim in enumerate(n_subdims):
        for j, gamma in enumerate(gammas):
            kernel_bases = [
                kn.kernel_subspace_bases(X_class, n_subdim, gamma) for X_class in X_train
            ]
            #print('kernel_basis', kernel_bases)
            kernel_bases = np.array(kernel_bases)
            #print(f"Kernel fit time: {fit_time}")

            kernel_similarities = [
                kn.kernel_similarity(_kernel_base, _X, X_test)
                for _kernel_base, _X in zip(kernel_bases, X_train)
            ]
            kernel_similarities = np.vstack(kernel_similarities).T
            #print('kernelsimilarities', kernel_similarities)

            # 正常クラスのみでanomalyscoreを出す
            pred = calc_anomaly_score(kernel_similarities[:, :(kernel_similarities.shape[1] - 1)])
            
            # AUCの計算
            _min = np.array(min(pred))
            _max = np.array(max(pred))

            re_scaled = (pred - _min) / (_max - _min)
            re_scaled = np.array(re_scaled, dtype=float)
            fpr, tpr, th = roc_curve(anomaly_labels, re_scaled)
            ftr_roc_auc[j, i] = auc(fpr, tpr)
            print(f"Subspace dimensions: {n_subdim}", f"Gamma: {gamma}", 'AUC', ftr_roc_auc[j, i])

            # plt.plot(fpr, tpr, marker='o')
            # plt.xlabel('FPR: False positive rate')
            # plt.ylabel('TPR: True positive rate')
            # plt.grid()
            # plt.savefig(knl_dir + 'ROCCurve_subdim' + str(n_subdim) + '_gamma' + str(gamma) + '.png')
            # plt.clf()

            
            df3 = pd.DataFrame(np.vstack([anomaly_labels, re_scaled]).T)
            df3.to_csv(knl_dir + 'auc_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv')
            df3 = pd.DataFrame(th)
            df3.to_csv(knl_dir + 'thresh_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv')

    df1 = pd.DataFrame(ftr_roc_auc)
    df1.to_csv(knl_dir + '0_auc_gamma_all.csv')

    #df1 = pd.DataFrame(ftr_roc_auc)
    #df1.to_csv(knl_dir + 'auc.csv')


def calc_kernel_subspace_bases(X_train, labels, X_test, y_test, anomaly_labels, knl_dir):
    n_subdims = range(1, 150, 1)
    gammas = range(1, 300, 10)

    df = pd.DataFrame(columns=["n_subdims", "gamma", "acc", "weighted_f1", "macro_f1", "fit_time", "predict_time"])
    ftr_roc_auc = np.zeros((-(-(300-1)//10), 150))

    for i, n_subdim in enumerate(n_subdims):
        print(f"Subspace dimensions: {n_subdim}")
        for j, gamma in enumerate(gammas):
            print(f"Gamma: {gamma}")
            start = time.time()
            kernel_bases = [
                kn.kernel_subspace_bases(X_class, n_subdim, gamma) for X_class in X_train
            ]
            #print('kernel_basis', kernel_bases)
            kernel_bases = np.array(kernel_bases)
            fit_time = time.time() - start
            print(f"Kernel fit time: {fit_time}")

            start = time.time()
            kernel_similarities = [
                kn.kernel_similarity(_kernel_base, _X, X_test)
                for _kernel_base, _X in zip(kernel_bases, X_train)
            ]
            kernel_similarities = np.vstack(kernel_similarities).T
            #print('kernelsimilarities', kernel_similarities)

            # 正常クラスのみでanomalyscoreを出す
            pred = calc_anomaly_score(kernel_similarities[:, :(kernel_similarities.shape[1] - 1)])
            
            # AUCの計算

            _min = np.array(min(pred))
            _max = np.array(max(pred))

            re_scaled = (pred - _min) / (_max - _min)
            re_scaled = np.array(re_scaled, dtype=float)
            fpr, tpr, _ = roc_curve(anomaly_labels, re_scaled)
            ftr_roc_auc[j, i] = auc(fpr, tpr)
            print('AUC', ftr_roc_auc[j, i])

            plt.plot(fpr, tpr, marker='o')
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.grid()
            plt.savefig(knl_dir + 'ROCCurve_subdim' + str(n_subdim) + '_gamma' + str(gamma) + '.png')
            
            df3 = pd.DataFrame(np.vstack([anomaly_labels, re_scaled]).T)
            df3.to_csv(knl_dir + 'auc_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv')

            predictions = fn.predict(kernel_similarities)
            predict_time = time.time() - start
            print(f"Predict time: {predict_time}")

            accuracy = accuracy_score(y_test, predictions)
            weighted_f1 = f1_score(y_test, predictions, labels=labels, average="weighted")
            macro_f1 = f1_score(y_test, predictions, labels=labels, average="macro")

            df = df.append({
                "n_subdims": n_subdim,
                "gamma": gamma,
                "acc": accuracy,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "fit_time": fit_time,
                "predict_time": predict_time,
            }, ignore_index=True)

            df.to_csv(knl_dir + "kernel.csv", index=False)

    df1 = pd.DataFrame(ftr_roc_auc)
    df1.to_csv(knl_dir + 'auc1.csv')
# %%
