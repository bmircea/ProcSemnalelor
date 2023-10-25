# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:00:39 2023

@author: Mircea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
#import sounddevice



def main():
    samples = np.linspace(-1, 1, 20000)
    sine = np.array([sinewave(n) for n in samples])
    cosine = np.array([cosinewave(n) for n in samples])
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(samples[:100], sine[:100])
    plt.title("Sinewave")
    plt.subplot(2, 1, 2)
    plt.plot(samples[:100], cosine[:100])
    plt.title("Cosinewave")
    
    plt.show()
    
    
def sinewave(n):
    amp = 3
    f = 800 #Hz
    phase = np.pi/2
    
    return amp * np.sin(2 * np.pi * f * n + phase)

def cosinewave(n):
    amp = 3
    f = 800 #Hz
    phase = 0
    
    return amp * np.cos(2 * np.pi * f * n + phase)

def main2():
    plt.figure(figsize=(24, 16))
    samples = np.linspace(-1, 1, 30000)    
    wave = lambda t, phase: np.cos(2 * np.pi * 800 * t + phase)
    
    n = np.random.normal(0, 1, 30000)
    i = 1;
    
    for p in (0, np.pi/2, 3*np.pi/2, np.pi):
        plt.subplot(2, 2, i)
        i+=1
        
        sinewave = np.array([wave(x, p) for x in samples])
        plt.plot(samples[:50], sinewave[:50])
        
        for snr in (0.1, 1., 10., 100.):
            # compute gamma
            gamma = np.sqrt(norm2(samples) / snr / norm2(n))
            print(gamma)
            
            
            
            sine_with_noise = sinewave + gamma * n
            plt.plot(samples[:50], sine_with_noise[:50])
            
    
    plt.show()
    
def norm2(x):
    s = np.sum([x1*x1 for x1 in x])
    return np.sqrt(s)

def main3():
    rate, x = scipy.io.wavfile.read('sine.wav')
    print(rate)
    print(x[:10])
    
    
def main4():
    # sine
    samples = np.linspace(-1, 1, 30000)
    sinewave = lambda x: 3 * np.sin(2 * np.pi * 800 * x)
    saw = lambda x: scipy.signal.sawtooth(2 * np.pi * 2000 * x)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(samples[:50], sinewave(samples)[:50])
    
    plt.subplot(3, 1, 2)
    plt.plot(samples[:50], saw(samples)[:50])
    
    plt.subplot(3, 1, 3)
    
    add = sinewave(samples) + saw(samples)
    plt.plot(samples[:50], add[:50])
    
    plt.show()
    

def main5():
    sinewave = lambda x, f: 3 * np.sin(2 * np.pi * f * x)
    
    samples = np.linspace(-1, 1, 30000)
    
    add = np.append([sinewave(x, 800) for x in samples], [sinewave(x, 1200) for x in samples])

    sounddevice.play(add, 44100)
    
    
def main6():
    plt.figure(figsize=(12, 8))
    #sines
    sinewave = lambda x, f: 3 * np.sin(2 * np.pi * f * x)
    
    samples = np.linspace(-1, 1, 20000)
    
    for i in (7500, 3750, 0):
        plt.plot(samples[:50], sinewave(samples, i)[:50], label=i)
        plt.legend()
        
    plt.show()
    
def main7():
    sinewave = lambda x: np.sin(2 * np.pi * 10 * x) 
    
    samples = np.linspace(-1, 1, 1000)
    sine1 = sinewave(samples)
    
    samples2 = np.array([samples[i] for i in range(3, len(samples), 4)])
    sine2 = sinewave(samples2)
    
    samples3 = np.array([samples[i] for i in range(4, len(samples), 4)])
    sine3 = sinewave(samples3)
    plt.figure(figsize=(12, 8))
    plt.plot(samples[:50], sine1[:50], label="initial")
    plt.plot(samples2[:50], sine2[:50], label="decimated - starting from 1st")
    plt.plot(samples3[:50], sine3[:50], label="decimated - starting from 2nd")
    plt.legend()
    plt.show()
    
    
def main8():
    samples = np.linspace(-np.pi/2, np.pi/2, 15000)
    f = 800 #Hz    
    sine = lambda x: np.sin(2*np.pi*f*x)
    taylor_a = lambda x: 2*np.pi*f*x
    pade_a = lambda x: (2*np.pi*f*x - (7*(2*np.pi*f*x)**3)/60)/(1 + ((2*np.pi*f*x)**2)/20)

    sinewave = sine(samples)
    taylor_a_wave = taylor_a(samples)
    pade_a_wave = pade_a(samples)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(samples, sinewave, label="true sine")
    plt.plot(samples[:50], taylor_a_wave[:50], label="Taylor approx")
    
    plt.subplot(2, 2, 2)
    plt.plot(samples[:50], sinewave[:50], label="true sine")
    plt.plot(samples[:50], pade_a_wave[:50], label="Pade approx")
    
    plt.subplot(2, 2, 3)
    plt.plot(samples[:50], (sinewave-taylor_a_wave)[:50], label="Error True sine - Taylor")
    
    plt.subplot(2, 2, 4)
    plt.plot(samples[:50], (sinewave-pade_a_wave)[:50], label="Error True sine - Pade")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main8()