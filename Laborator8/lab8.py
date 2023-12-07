# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:03:40 2023

@author: Mircea
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

N = 1000

def main1() -> np.array:

    samples = np.linspace(0, 5, N)
    # wave = A * sin(2 * pi * f * t + A)
    seasonal_wave = lambda t : 0.2 * np.sin(2 * np.pi * 4 * t) + 0.4 * np.cos(2 * np.pi * 6 * t)
    seasonal = np.array([seasonal_wave(x) for x in samples])
    trend_eq = lambda x : x**2 + 0.001*x + 0.1
    trend =  np.array([trend_eq(x) for x in samples])
    noised = np.random.normal(0, 0.1, size=N)
    
    ts_data = trend + seasonal + noised
    
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(trend)
    plt.title("Trend")

    plt.subplot(4, 1, 2)
    plt.plot(seasonal)
    plt.title("Seasonal")

    plt.subplot(4, 1, 3)
    plt.plot(noised)
    plt.title("Variations")

    plt.subplot(4, 1, 4)
    plt.plot(ts_data)
    plt.title("Original data")
    
    plt.show()    


    return ts_data


def main2(timeseries) -> np.array:
    # Calculam vectorul de autocorelatie cu np.correlate
    
    corr = np.correlate(timeseries, timeseries, mode='same')
    
    # print(corr)
    
    # cu modul same, vom primi vectorul de corelatie + primul element aditional
    # dar elementul cel mai mare este cel de pe pozitia 2
    # deci
    corr = corr[1:]
    
    print(corr)
    plt.figure(figsize=(12, 8))
    plt.plot(corr/np.linalg.norm(corr))
    plt.title("Vectorul de autocorelatie")
    plt.show()
    
    return corr

def pacf(timeseries):
    
    # p = 3? 
    from statsmodels.graphics.tsaplots import plot_pacf
    pacf = plot_pacf(timeseries, lags=25)


def main3(timeseries):
    # Train & test
    
    train_data = timeseries[:len(timeseries)-100]
    test_data = timeseries[len(timeseries)-100:]
    
    ar_model = AutoReg(train_data, lags=3).fit()
    
    print(ar_model.summary())
    
    pred = ar_model.predict(start=len(train_data), end=N-1, dynamic=False)
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_data, label="Dataset")
    plt.plot([x for x in range(len(train_data), N)], pred, color="green", label="Prediction")
    plt.plot([x for x in range(len(train_data), N)], test_data, color="red", label="Actual value")
    plt.legend()
    
    plt.show()
    

if __name__ == "__main__":
    timeseries = main1()
    corr = main2(timeseries)
    main3(timeseries)
        
    
    
    