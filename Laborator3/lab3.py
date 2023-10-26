# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:30:17 2023

@author: Mircea
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def main1():
    F = []
    N = 8
    for i in range(1, N):
        f = [1 + 0j]
        for j in range(1, N):
            f.append(math.e ** (2 * np.pi * 1j * i * j / N))
        F.append(f)
        
    plt.figure(figsize=(12, 8))
    plt.subplot(8, 1, 1)
    
    
    print(F[1])


if __name__ == "__main__":
    main1()