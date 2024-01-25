# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:06:45 2024

@author: Mircea
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd

def main1(mean, sigma):
    # 1-D
    distrib = np.random.normal(0, 1, 10000)
    
    dist = [x * np.sqrt(sigma) + mean for x in distrib]
    
    counts, bins = np.histogram(dist)
    
    plt.stairs(counts, bins)
    
    plt.show()
    
    
    # 2-D
    m = [0, 0]
    sig = [[1, 0.6], [0.6, 2]]
    
    U, S, Vh = np.linalg.svd(sig)
    
    z = np.random.randn(2)
    
    samples = mean + np.dot(U, np.dot(np.sqrt(S), z))
    
    
    
def nd(mean, sigma, x):
    return math.exp(-0.5 * ((x-mean) / sigma )**2) / math.sqrt(2*np.pi*sigma)

def main2():
    # liniar
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    
    covm = [[_x * _y for _y in y] for _x in x]
    mean = [0 for _x in x]

    U, S, Vh = np.linalg.svd(covm)
    
    z = np.dot(U, np.dot(np.sqrt(S), x))
    
    plt.plot(x, z)
    plt.show()

    
    
def exponential_quadratic_kernel(x1, x2):
    l = 50.0
    expf = lambda xi, xj, l: math.exp(-(abs(xi-xj)**2 / (2*l**2)))
    covm = [[(50.0 ** 2) * expf(xi, xj, l) for xj in x2] for xi in x1]
    return covm
    
def main3():
    data = pd.read_csv('co2_daily_mlo.csv', header=None, names=['year', 'month', 'day', 'e', 'co2'])
    
    g_d = data.groupby(['year', 'month'])
    
    result = g_d.agg({
        'year': 'first',
        'month': 'first',
        'co2': 'mean'
    })
    
    result['months'] = (result['year'] - result.values[0][0])*12 + result['month'] - result.values[0][1]
    
    lin = ((result['months'] - result['months'].mean()) * (result['co2'] - result['co2'].mean())).sum() / ((result['months'] - result['months'].mean())** 2).sum()  
    
    lin2 = result['co2'].mean() - lin * result['months'].mean()
    
    
    data['co2'] = data['co2']-lin2
    
    data["date"] = pd.to_datetime(data[["year", "month", "day"]])
    
    data = data[["date", "co2"]].set_index("date")
    
    data.plot()
    
    X = (data.index.year * 12 + data.index.month).to_numpy()
    X = X[X > 12 * 2023]
    X = X - 23693
    
    print(len(X))
    y = data[data.index.year >= 2023]["co2"].to_numpy()
    print(len(y))
    
    X_test = np.arange(start=584, stop=587, step=1)
    X_test = np.array([[x] * 30 for x in X_test]).reshape((1, -1))
    
    kernel =  exponential_quadratic_kernel(X, X)
    print(kernel)
    L = np.linalg.cholesky(kernel)
    
    kernel2 = exponential_quadratic_kernel(X, X_test)
    
    L2 = np.linalg.solve(L, kernel2)
    m = np.dot(L2.T, np.linalg.solve(L, y))
    
    kernel3 = exponential_quadratic_kernel(X_test, X_test)
    
    s2 = np.diag(kernel3) - np.sum(L2**2, axis=0)
    
    sdev = np.sqrt(s2)
    
    
    plt.plot(X, y, linestyle="dashed")
    plt.plot(X_test, m, color="tab:blue", alpha=0.4)
    plt.fill_between(X_test.ravel(), s2 - sdev, s2 + sdev, color="tab:blue", alpha=0.2)
    
    
    
    
    

    

if __name__ == "__main__":
    main3()