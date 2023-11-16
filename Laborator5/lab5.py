# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:10:17 2023

@author: Mircea
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

def main():
    # a) fs = 1/3600 Hz
    # b) Esantioanele acopera 726 de zile (18828 ore), adica aproximativ 2 ani
    # c) Frecventa maxima este 120Hz
    
    data = []
    with open('Train.csv') as file:
        lines = [line for line in file]
        lines.pop(0) # Remove header
        split_data = lambda split_line : (split_line[0], datetime.datetime.strptime(split_line[1], "%d-%m-%Y %H:%M"), split_line[2])
        for line in lines:
            data.append(split_data(line.split(',')))
    
    plt.figure()
    
    freqValues = [entry[2] for entry in data]    
    plt.plot([r[0] for r in data][:1000], freqValues[:1000])    
    plt.show()
    X = np.fft.fft(freqValues) # Calcul FFT
    X = abs(X/len(freqValues))
    X = X[:len(freqValues)//2]
    print(X)
    
    Fs = 20 # frecventa de esantionare    
    f = np.linspace(0, len(freqValues)//2, len(freqValues)//2) # Use int division
    f = (Fs * f)/len(freqValues)
    
    plt.figure()
    plt.plot(f, X)
    plt.show()            
   
    
   # e) 
   # f) Principalele componente de frecventa sunt la 0Hz, 1Hz, 0.2Hz, 1.7Hz
   # g) 
   
    for i in range(1000, len(data)):
        if (data[i][1].weekday() == 0): # Daca e zi de luni
            plt.figure(figsize=(24, 16))
            idx = [r[0] for r in data[i:i+720]]
            vals = [r[2] for r in data[i:i+720]]
            plt.plot(idx, vals)
            plt.show()
            break
       
       




if __name__ == "__main__":
    main()