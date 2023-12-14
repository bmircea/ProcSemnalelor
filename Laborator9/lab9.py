"""

Laborator 9


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
    
    """
    plt.figure(figsize=(12, 8))
    plt.subplot(5, 1, 1)
    plt.plot(trend)
    plt.title("Trend")

    plt.subplot(5, 1, 2)
    plt.plot(seasonal)
    plt.title("Seasonal")

    plt.subplot(5, 1, 3)
    plt.plot(noised)
    plt.title("Variations")

    plt.subplot(5, 1, 4)
    plt.plot(ts_data)
    plt.title("Original data")
    
    plt.show()    
    """

    return ts_data

def expMA(timeseries, alpha):
    # s[0] = x[0]
    # s[t] = alpha*x[t] + (1-alpha) * s[t-1]
    
    s = []
    
    s.append(timeseries[0])
    
    for i in range(1, len(timeseries)):
        s.append(alpha*timeseries[i] + (1-alpha) * s[i-1])
    
    """
    plt.subplot(5, 1, 5)
    plt.plot(s)
    plt.title("TS Generated with ExpMA")
    
    plt.show()
    """
    return s
    

def findAlpha(timeseries):
    vals = np.linspace(0, 1, 100)
    err = []
    
    for v in vals:
        s = expMA(timeseries, v)
        err.append(np.subtract(timeseries, s))
        
    
    
    meanerrs = []
    
    for e in err:
        error = 0.
        for value in e:
            error += abs(value)
            
        meanerrs.append(error/len(timeseries))


            
        
        
    
    minavgerr = 1.
    alpha = 1.
    
    meanerrs.pop()
    mine = np.min(meanerrs)
    print(meanerrs)    
    print(np.where(meanerrs == mine))
    
    """
    for e in range(100):
        if np.mean(e[0]) < minavgerr:
            minavgerr = np.mean(e[0])
            alpha = e[1]        
    
    print(alpha, minavgerr)
    """
    
def MA(timeseries, p):
    eps = np.random.normal(0, 1, p+2)
    


if __name__ == "__main__":
    timeseries = main1()
    #expMA(timeseries, 0.65)
    findAlpha(timeseries)
    
    