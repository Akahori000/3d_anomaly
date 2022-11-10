import os
import glob
import datetime
import numpy as np
import shutil

testn = {'bookshelf':'180', 'laptop':'184', 'knife':'170', 'train':'156', 'motorbike':'134', 'guitar':'320', 'faucet':'298'} # test のとき
valn = {'bookshelf':'90', 'laptop':'92', 'knife':'84', 'train':'78', 'motorbike':'68', 'guitar':'160', 'faucet':'150'} # val のとき
names = ['bookshelf', 'laptop', 'knife', 'train', 'motorbike', 'guitar', 'faucet']
types = ['AE_', '1_']
epochs = ["150", "200", "250", "299"]
testways = ['kernel_pred2', 'pred_NA_halfhalf2']
datasets = [testn, valn]

for type in types:
    for i in names:
        for epoch in epochs:
            for testway in testways:
                for dataset in datasets:
                    #path = './data/calculated_features_random/model' + type + i + '/both_features/' + 'c_epoc_' + epoch + '_data' + dataset[i] + '/' + testway + '/001*'
                    paths = ['./data/objset2/calculated_features/model' + type + i + '/both_features/' + 'c_epoc_' + epoch + '_data' + dataset[i] + '/' + testway + '/001*',
                            './data/objset2/calculated_features/model' + type + i + '/both_features/' + 'c_epoc_' + epoch + '_data' + dataset[i] + '/' + testway + '/002*',
                            './data/objset2/calculated_features/model' + type + i + '/both_features/' + 'c_epoc_' + epoch + '_data' + dataset[i] + '/' + testway + '/003*']
                    for path in paths:
                        #print(path)
                        files = glob.glob(path)
                        #print(len(files))
                        a = np.zeros(len(files))
                        for cnt, j in enumerate(files):
                            t = os.path.getctime(j)
                            #print(datetime.datetime.fromtimestamp(t))
                            a[cnt] = t
                        
                        if len(files) == 2:
                            print(path)
                            #print(a)
                            #print(a[0]-a[1])
                            if a[0] - a[1] > 0:
                                print('delete', datetime.datetime.fromtimestamp(os.path.getctime(files[1])) )
                                os.remove(files[1])
                                print('keep', datetime.datetime.fromtimestamp(os.path.getctime(files[0])))
                            else:
                                print('delete', datetime.datetime.fromtimestamp(os.path.getctime(files[0])) )
                                os.remove(files[0])
                                print('keep', datetime.datetime.fromtimestamp(os.path.getctime(files[1])))
                        elif len(files) > 2:
                            print(path)
                            for t in files:
                                print('keep', datetime.datetime.fromtimestamp(os.path.getctime(t)), t)
                            print("2ijou")
                        #else:
                            #print('file=1', datetime.datetime.fromtimestamp(os.path.getctime(files[0])))
                            


                            #dt = datetime.datetime.fromtimestamp(p.stat().st_ctime)
                