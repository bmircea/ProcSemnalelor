# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:11:14 2023

@author: Mircea
"""
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

def main1():
    samples = np.linspace(-1, 1, 10000)
    wave1 = lambda t : np.sin(2 * np.pi * 10 * t) + 1/2 * np.sin(2 * np.pi * 12 * t + 3*np.pi/4) + 1/4 * np.sin(2 * np.pi * 8 * t + np.pi/2)

    f = lambda m , wave : np.sum([wave[n] * math.e ** (-2 * np.pi * 1j * n * m / len(samples)) for n in range(len(wave))])
    
    N = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    wave = np.array([wave1(s) for s in samples])
    times = []
    for dim in N:
        start_time = time.time_ns()
        for i in range(dim):
            f(i, wave[:dim])
            
        times.append(time.time_ns() - start_time)
        
        start_time = time.time_ns()
        np.fft.fft(wave[:dim])
        times.append(time.time_ns() - start_time)
        
    times_s = [t / 10**9 for t in times]
    
    print(times_s)
    
    plt.figure(figsize=(12, 8))
        
    for n in range(len(N)):
        plt.stem(N[n], times_s[2*n], linefmt='C1-')
        plt.stem(N[n], times_s[2*n+1], linefmt='C2-')
        
    plt.xlabel('Elemente')
    plt.ylabel('Timp')
    
    plt.show()


def main2():
    samples = np.linspace(-1, 1, 100)
    sine_function = lambda t, f: np.sin(2 * np.pi * f * t)
    
    sinewave1 = [sine_function(t, 55) for t in samples]
    sinewave2 = [sine_function(t, 355) for t in samples]
    sinewave3 = [sine_function(t, 655) for t in samples]
    
    plt.figure(figsize=(20, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(samples[15], sinewave1[15], c='r')
    plt.scatter(samples[50], sinewave1[50], c='r')
    plt.scatter(samples[70], sinewave1[70], c='r')
    plt.scatter(samples[95], sinewave1[95], c='r')
    plt.plot(samples, sinewave1)
    plt.subplot(3, 1, 2)
    plt.plot(samples, sinewave2)
    plt.scatter(samples[15], sinewave2[15], c='r')
    plt.scatter(samples[50], sinewave2[50], c='r')
    plt.scatter(samples[70], sinewave2[70], c='r')
    plt.scatter(samples[95], sinewave2[95], c='r')
    plt.subplot(3, 1, 3)
    plt.scatter(samples[15], sinewave3[15], c='r')
    plt.scatter(samples[50], sinewave3[50], c='r')
    plt.scatter(samples[70], sinewave3[70], c='r')
    plt.scatter(samples[95], sinewave3[95], c='r')
    plt.plot(samples, sinewave3)
    
    
    
    plt.show()
    
def main3():
    samples = np.linspace(-1, 1, 1500)
    sine_function = lambda t, f: np.sin(2 * np.pi * f * t)
    
    sinewave1 = [sine_function(t, 55) for t in samples]
    sinewave2 = [sine_function(t, 355) for t in samples]
    sinewave3 = [sine_function(t, 655) for t in samples]
    
    plt.figure(figsize=(20, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(samples[15], sinewave1[15], c='r')
    plt.scatter(samples[50], sinewave1[50], c='r')
    plt.scatter(samples[70], sinewave1[70], c='r')
    plt.scatter(samples[95], sinewave1[95], c='r')
    plt.plot(samples[:150], sinewave1[:150])
    plt.subplot(3, 1, 2)
    plt.plot(samples[:400], sinewave2[:400])
    plt.scatter(samples[15], sinewave2[15], c='r')
    plt.scatter(samples[50], sinewave2[50], c='r')
    plt.scatter(samples[70], sinewave2[70], c='r')
    plt.scatter(samples[95], sinewave2[95], c='r')
    plt.subplot(3, 1, 3)
    plt.scatter(samples[15], sinewave3[15], c='r')
    plt.scatter(samples[50], sinewave3[50], c='r')
    plt.scatter(samples[70], sinewave3[70], c='r')
    plt.scatter(samples[95], sinewave3[95], c='r')
    plt.plot(samples[:800], sinewave3[:800])
    
    
    
    plt.show()
    
def ex4():
    # Frecventa maxima emisa de instrument este 200Hz
    # deci, ca semnalul discretizat sa contina toate componentele de frecventa
    # ale instrumentului, semnalul band pass trebuie esantionat 
    # cu minim 2*200Hz = 400Hz
    pass

def main5():
    # https://imgur.com/a/IqTcsTt
    pass

def main6():
    # Citire semnal
    rate, x = scipy.io.wavfile.read('voc.wav')
    
    # Impartire in grupuri [(i+1) * hgs, (i+3) * hgs ]
    groups = []
    group_size = int(len(x) / 100)
    
    for i in range(99):
        groups.append(np.array([x[j] for j in range((int)(group_size/2*(2*i+1)), (int)(group_size/2*(2*i+3)))]))
    
    hgs = (int)(group_size/2)
    groups.append(np.concatenate((x[hgs:], x[:hgs])))
    
    # FFT
    ffts = []
    for g in groups:
        ffts.append(np.fft.fft(g))
        
    #print(ffts)
    
    #for f in ffts:
     #   f = f.reshape(, -1)
    
    #print(ffts[0].shape)
    M = np.array(np.asmatrix(ffts[0]))
    for i in range(1, len(ffts)):
        M = np.append(M, np.asmatrix(ffts[i]), 1)
        
    #M.reshape(-1, 3840)
    #print(M.shape)
    #print(M)
    
    #M = np.transpose(M)
    #print(M.shape)
    
    print(group_size)
    plt.figure(figsize=(12, 8))
    plt.specgram(x, group_size, noverlap=int(group_size/2))
    
    plt.show()

def ex7():
    # Puterea zgomotului = Psemnal / SNR
    # SNR = 10 ** (SNRdb/10)
    # deci Pzgomot = 90/10**8 = 9/10**7 db
    pass


if __name__ == '__main__':
    main6()