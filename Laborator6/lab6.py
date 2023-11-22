# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:23:41 2023

@author: Mircea
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import datetime

def main1():
    x = np.random.rand(100)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x)
    
    for i in range(3):
        x = np.convolve(x, x, mode='full')
        plt.subplot(2, 2, i+2)
        plt.plot(x)
        plt.title('Op ' + str(i+1))
        
    plt.show()
    print(x)
    
def main2(p, q):
    
    prod = np.convolve(p, q, mode='full')
    
    P = np.fft.fft(p)
    Q = np.fft.fft(q)
    
    R = np.multiply(P, Q)
    print(R)
    r = np.fft.ifft(R)
    
    print(prod)
    print(r)
    
# 3
def drept(Nw):
    # w(n) = 1
    return Nw * [1]
    
def hanning(Nw):
    # w(n) = 0.5[1 - cos(2*pi*n/N)]
    w = []
    
    for i in range(Nw):
        w.append(0.5*(1 - np.cos(2 * np.pi * i / Nw)))
        
    return w

def main3():
    sine = lambda A, f, x, phase : A + np.sin(2 * np.pi * f * x + phase)
    
    samples = np.linspace(0, 1, 400)
    
    sinewave = [sine(1, 100, x, 0) for x in samples]
    
    filtered_1 = np.multiply(sinewave, drept(200))
    
    filtered_2 = np.multiply(sinewave, hanning(200))
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(samples, sinewave)
    plt.subplot(3, 1, 2)
    plt.plot(samples, filtered_1)
    plt.subplot(3, 1, 3)
    plt.plot(samples, filtered_2)
    
    plt.show()
    
def main4():
    data = []
    with open('Train.csv') as file:
        lines = [line for line in file]
        lines.pop(0) #Remove header
        split_data = lambda split_line : (split_line[0], datetime.datetime.strptime(split_line[1], "%d-%m-%Y %H:%M"), split_line[2].replace('\n', ''))
        for line in lines:
            data.append(split_data(line.split(',')))
            
    start_day = data[0][1].day
    i = 0
    while data[i][1].day == start_day:
        i += 1
   
    # a) date pe 3 zile
    data_selection = [data[j] for j in range(i, i+72)]
   
    
    # b) filtru moving average
    filtru_ma = lambda x, w : np.convolve(x, np.ones(w)/w, 'valid')
    
    x = [int(l[2]) for l in data_selection]
    
    plt.figure(figsize=(18, 12))
    plt.plot(x, label='Semnalul nefiltrat')
    w_vals = [5, 11, 14]
    for i in range(len(w_vals)):
        flt = filtru_ma(x, w_vals[i])
       # plt.plot(flt, label='Filtru w='+str(w_vals[i]))
    
    #plt.legend()
    
    #plt.show()
    
    
    # c)
    # Pentru filtrul low pass putem alege o frecventa de cutoff de 6Hz
    # Normalizata, valoarea frecventei de cutoff este 0.375, cu frecventa Nyquist 16Hz
    
    Wn = 0.375 # frecventa de cutoff normalizata
    N = 5 # Ordinul filtrului
    rp = 5 #dB - atenuarea ondulatiilor
    # d)
    b, a = scipy.signal.butter(N, Wn, btype='low')
    butter_f = (b, a)
    b, a = scipy.signal.cheby1(N, rp, Wn, btype='low')
    chebyshev_f = (b, a)
    
    # e)
    y_butter = scipy.signal.filtfilt(butter_f[0], butter_f[1], x)
    y_cheby = scipy.signal.filtfilt(chebyshev_f[0], chebyshev_f[1], x)
    
    plt.plot(y_butter, label='Butterworth N=5')
    plt.plot(y_cheby, label='Chebyshev N=5')
    
    
    #plt.legend()
    
    #plt.show()
    
    # Filtrul butterworth are o filtrare mai putin agresiva, in timp ce 
    # filtrul Chebyshev netezeste mai mult semnalul
    
    # f)
    
    b, a = scipy.signal.butter(N-2, Wn, btype='low')
    plt.plot(scipy.signal.filtfilt(b, a, x), label='Butterworth N=3')
    b, a = scipy.signal.butter(N+2, Wn, btype='low')
    plt.plot(scipy.signal.filtfilt(b, a, x), label='Butterworth N=7')
    
    b, a = scipy.signal.cheby1(N-2, rp-1, Wn, btype='low')
    plt.plot(scipy.signal.filtfilt(b, a, x), label='Cheby N = 3, rp = 4')
        
    b, a = scipy.signal.cheby1(N+2, rp-1, Wn, btype='low')
    plt.plot(scipy.signal.filtfilt(b, a, x), label='Cheby N = 7, rp = 4')
    
    b, a = scipy.signal.cheby1(N-2, rp+1, Wn, btype='low')
    plt.plot(scipy.signal.filtfilt(b, a, x), label='Cheby N = 3, rp = 6')
        
    b, a = scipy.signal.cheby1(N+2, rp+1, Wn, btype='low')
    plt.plot(scipy.signal.filtfilt(b, a, x), label='Cheby N = 7, rp = 6')
    
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    #main2(np.array([1., 1., 1.]), np.array([0.5, 0.2, 0.3]))
    #print(hanning(10))
    main4()