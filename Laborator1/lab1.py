import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math



def main():
    lin = np.linspace(0., 0.3, 600)
    
    
    
    # plot
    
    fig, axs = plt.subplots(3)
    
    fig.suptitle("Plot")
    
    axs[0].plot(lin, [x(v) for v in lin])
    axs[1].plot(lin, [y(v) for v in lin])
    axs[2].plot(lin, [z(v) for v in lin])
    
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x(t)")
    
    axs[1].set_xlabel("t") 
    axs[1].set_ylabel("y(t)")
    
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("z(t)")
    
    
    #plt.show()
    
    # esantionare
    fig2, axs2 = plt.subplots(3)
        
    v = []
    i = 0;
    while (i < 200):
        v.append(np.random.random_sample())
        i += 1
        
    
    v.sort()
    
    axs[0].stem(v[:50], [x(t) for t in v[:50]])
    axs[1].stem(v[:50], [y(t) for t in v[:50]])
    axs[2].stem(v[:50], [z(t) for t in v[:50]])
    
    

   
    plt.show()
    
    
def main2():
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    
    # semnal 400Hz(frecventa semnalului, NU DE ESANTIONARE), 1600 esantioane - 1 sec
    # aflam x(T)
    samples = np.linspace(0, 1, 1600, endpoint=False)
    sine = np.sin(2 * np.pi * 400 * samples)
    
    plt.title('Sine wave 400Hz')
    plt.stem(samples[:50], sine[:50], 'yo')
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.legend(loc='upper right')
    
    #samples = np.linspace(0, T, n_samples, endpoint=False)
    #signal = Amplitude * np.sin(2 * np.pi * sign_freq * samples)


    # semnal 800Hz, 3 sec    
    sine2_samples = np.linspace(0, 3, 500, endpoint=False)
    sine2_signal = np.sin(2 * np.pi * 800 * sine2_samples)
    plt.subplot(2, 2, 2)
    plt.title('Sine wave 800Hz')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.plot(sine2_samples[:50], sine2_signal[:50])
    
    # sawtooth, f=240Hz
    saw_samples = np.linspace(0, 1, 500, endpoint=False)
    saw_signal = sawtooth(saw_samples)
    plt.subplot(2, 2, 3)
    plt.plot(saw_samples, saw_signal)
    
    
    # square, f=300Hz
    square_samples = np.linspace(0, 5, 400, endpoint=False)
    square_signal = np.sign(np.sin(2 * np.pi * 300 * square_samples))
    print(square_signal)
    plt.subplot(2, 2, 4)
    plt.plot(square_samples[:50], square_signal[:50])
    
    # 2D random
    #rand2d = np.array(np.zeros((128, 128)))
    rand2d = np.random.rand(128, 128)
    
    plt.figure(2)
    
    plt.imshow(rand2d)
    
    
        
    
    
def sawtooth(x):
    return x - np.floor(x)
        
    
    
    
def x(t : float):
    return math.cos(520. * np.pi * t + np.pi / 3.)

def y(t : float):
    return math.cos(280. * np.pi * t - np.pi / 3.)

def z(t: float):
    return math.cos(120. * np.pi * t  + np.pi / 3.)

# Ex 3:
# a) fs = 2000Hz = 1/T => T = 0.5 ms
#
#
# b) 2000 esantioane/secunda = > 7.2M esantioane/ora
#   
#    7.2M * 4b / 1B = 3.6 MB
#




if __name__ == "__main__":
    main2()