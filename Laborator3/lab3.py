# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:30:17 2023

@author: Mircea
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def main1():
    N = 4
    F = fourier_m(N)
        
    real = lambda i, V : [f.real for f in V[i]]
    imag = lambda i, V : [f.imag for f in V[i]]
        
    plt.figure(figsize=(12, 8))
   
    for i in range(len(F)):
        plt.subplot(len(F), 1, i+1)
        plt.stem(real(i, F), imag(i, F))
    
    plt.show()
    
    # matricea este unitara <=> complexa si ortogonala
    
    # ortogonala <=> transpusa (dot) inversa = Identitatea
    print(F)
    
    A = np.matrix(F)
    print(A)
    AH = A.getH() # transpusa
    print(AH)
    print('--')
    print(np.identity(N))
    print(np.dot(AH, A))
    if (np.allclose(np.dot(AH, A), np.identity(N))):
        print('Matricea Fourier este unitara')

def fourier_m(N):
    F = []
    for i in range(N):
        f = []
        for j in range(N):
            f.append(math.e ** (-2* np.pi * 1j * i * j / N))
        F.append(f)
        
    return F
    

def main2():
    samples = np.linspace(-1, 1, 1000)
    wave = lambda n, phase: np.sin(2 * np.pi * 3 * n + phase)
    c = lambda n : wave1[n] * math.e ** (-2 * np.pi * 1j * n/1000) 
    z = lambda n, w : wave1[n] * math.e ** (-2 * np.pi * 1j * w * n/1000)
    wave1 = [wave(n, 0) for n in samples]
    y = [c(n) for n in range(1, len(wave1))]
    
    #plt.figure(figsize=(30, 10))
    fig = plt.subplot(3, 2, 1)
    plt.plot(samples, wave1)
    plt.xlabel('Timp(esantioane)')
    plt.ylabel('Amplitudine')
    plt.stem(samples[550], wave1[550], linefmt='C1-')
    #plt.ylim((0, 1000))
    #plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    fig = plt.subplot(3, 2, 2)
    plt.plot([f.real for f in y], [f.imag for f in y])
    plt.stem([f.real for f in y][550], [f.imag for f in y][550], linefmt='C1-', orientation='vertical', bottom=0) 
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlabel('Real')
    plt.ylabel('Imaginar')
    
    i = 3
    for w in [1, 3, 5, 7]:
        plt.subplot(3, 2, i)
        plt.title('w = ' + str(w))
        plt.xlabel('Real')
        plt.ylabel('Imaginar')
        zz = [z(n, w) for n in range(len(wave1))]
        plt.plot([f.real for f in zz], [f.imag for f in zz])
        i += 1
    
    
    plt.show()
    
    
def main3():
    samples = np.linspace(-1, 1, 1000)
    wave1 = lambda t : np.sin(2 * np.pi * 10 * t) + 1/2 * np.sin(2 * np.pi * 12 * t + 3*np.pi/4) + 1/4 * np.sin(2 * np.pi * 8 * t + np.pi/2)
    
    x = [wave1(s) for s in samples]
    f = lambda w , wave : np.sum([wave[n] * math.e ** (-2 * np.pi * 1j * n * w / len(samples)) for n in range(len(samples))])
    
    X = []
    W = [0, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 60, 70, 90, 120]
    for w in W:
        X.append(f(w, x))      
        
    plt.subplot(1, 2, 1)
    plt.plot(samples, x)

    plt.subplot(1, 2, 2)
    for i in range(len(W)):
        plt.stem(W[i], abs(X[i]))
    plt.show()    
        
    print([abs(xx) for xx in X])
    
if __name__ == "__main__":
    main1()