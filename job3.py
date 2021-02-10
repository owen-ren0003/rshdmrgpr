# INSTRUCTIONS TO RUN: Put this file along with rs_hdmr_gpr.py, KEDdataset.dat in the SAME DIRECTORY 
# and run your Bash script.

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from rs_hdmr_gpr import *


# Prints the RMSE
def get_RMSE(y, y_pred):
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    print(f'The RMSE is {rmse}')
    return rmse


if __name__ == '__main__':
    print(os.getcwd())
    # Synthetic DataSet

    # Physics DataSet
    # Extracts the data set
    columns = []
    for i in range(7):
        columns.append(f'a{i + 1}')
    columns.append('out')
    data = pd.read_csv('KEDdataset.dat', sep='\s+', names=columns)

    # Scales the data set to be between [0, 1]
    scale = data['out'].max() - data['out'].min()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    data_scaled = pd.DataFrame(scaler.transform(data), columns=columns)
    features = data_scaled.drop(columns=['out'])
    labels = data_scaled['out']

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=5000, test_size=None, random_state=42)
    d = data.shape[1] - 1

    matrices1, kernels1 = kernel_matrices(1, 7, 0.5)
    matrices2, kernels2 = kernel_matrices(2, 7, 0.5)
    matrices3, kernels3 = kernel_matrices(3, 7, 0.5)
    matrices4, kernels4 = kernel_matrices(4, 7, 0.5)
    matrices5, kernels5 = kernel_matrices(5, 7, 0.5)
    matrices6, kernels6 = kernel_matrices(6, 7, 0.5)
    matrices7, kernels7 = kernel_matrices(7, 7, 0.5)

    # Initializes the Model classes for training
    hdmr_1d = RSHDMRGPR(len(matrices1), matrices1, kernels1)
    hdmr_2d = RSHDMRGPR(len(matrices2), matrices2, kernels2)
    hdmr_3d = RSHDMRGPR(len(matrices3), matrices3, kernels3)
    hdmr_4d = RSHDMRGPR(len(matrices4), matrices4, kernels4)
    hdmr_5d = RSHDMRGPR(len(matrices5), matrices5, kernels5)
    hdmr_6d = RSHDMRGPR(len(matrices6), matrices6, kernels6)
    hdmr_7d = RSHDMRGPR(len(matrices7), matrices7, kernels7)

    hdmr = [hdmr_1d, hdmr_2d, hdmr_3d, hdmr_4d, hdmr_5d, hdmr_6d, hdmr_7d]
    alphas = [3 * 1e-3, 8 * 1e-4, 3 * 1e-4, 8 * 1e-5, 3 * 1e-5, 8 * 1e-6, 3 * 1e-6]

    y = [y_train]
    preds = []
    RMSEs = []
    for i in range(len(hdmr)):
        hdmr[i].train(x_train, y[i], alphas=alphas[i], cycles=50, optimizer="fmin_l_bfgs_b", opt_every=5, scale_down=(0.2, 2))

        y_pred = batch_predict(hdmr[i], data_scaled.drop(columns=['out']))
        preds.append(y_pred.copy())
        for j in range(i):
            y_pred += preds[j]
        rmse = get_RMSE(y_pred * scale, data_scaled['out'] * scale)
        RMSEs.append(rmse)

        y_pred1 = hdmr[i].predict(x_train)
        y.append(y[i] - y_pred1)

    print(RMSEs)
    models1 = hdmr[0].get_models()
    models2 = hdmr[1].get_models()
    models3 = hdmr[2].get_models()
    models4 = hdmr[3].get_models()
    models5 = hdmr[4].get_models()
    models6 = hdmr[5].get_models()
    models7 = hdmr[6].get_models()
    models = [models1, models2, models3, models4, models5, models6, models7]

    res = pd.DataFrame()
    for i in range(len(models)):
        j = 0
        for a in combinations([0, 1, 2, 3, 4, 5, 6], i + 1):
            name = ""
            for x in a:
                name += f'{x}_'
            res[name] = batch_predict(models[i][j], x_train.iloc[:, list(a)])
            j += 1

    for i in range(len(models)):
        for j in range(len(models[i])):
            print(models[i][j].kernel_)

    res.std().to_csv('job3_std.csv')
