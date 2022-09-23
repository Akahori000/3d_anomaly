# %%
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

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


def calc_kernel_subspace_bases(X_train, labels, X_test, y_test, anomaly_labels, test_dir):
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
            anomaly_score = calc_anomaly_score(kernel_similarities[:, :(kernel_similarities.shape[1] - 1)])
            # AUCの計算
            fpr, tpr, _ = roc_curve(anomaly_labels, anomaly_score)
            ftr_roc_auc[j, i] = auc(fpr, tpr)
            print('AUC', ftr_roc_auc[j, i])
            df3 = pd.DataFrame(np.vstack([anomaly_labels, anomaly_score]).T)
            df3.to_csv(test_dir + 'kernel_pred/' + 'auc_subdim' + str(n_subdim) + '_gamma' + str(gamma) +  '.csv')

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

            df.to_csv(test_dir + 'kernel_pred/' + "kernel.csv", index=False)

    df1 = pd.DataFrame(ftr_roc_auc)
    df1.to_csv(test_dir + 'kernel_pred/'  + 'auc.csv')