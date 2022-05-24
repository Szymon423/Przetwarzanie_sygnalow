from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

filename = input('Podaj ścieżkę do pliku .wav: ')

try:
    samplerate, data = wavfile.read("Samples\\" + filename + ".wav")
except (NameError, TypeError) as error:
    print(error)
else:
    print(data[:, 0])
    print(data[:, 1])
    print(samplerate)

    # czas trwania sygnału
    time = data.shape[0] / samplerate #
    print(data.shape[0])
    x = np.linspace(0,time,data.shape[0])
    print(data.shape)
    print(time)

    # przebieg czasowy
    plt.figure(1)
    plt.plot(x, data[:, 0])
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')

    # widmo amplitudowe
    b = np.array([(ele/2**13.)*2-1 for ele in data])
    c = fft(b[:, 1])
    d = len(c / 2)
    ff = np.array(range(data.shape[0]//2))
    e = ff/data.shape[0]*samplerate

    plt.figure(2)
    plt.semilogy(e[10:], abs(c[:d])[10:data.shape[0]//2], 'r')
    plt.ylim(10**2, 10**5)
    plt.xlabel('f [Hz]')
    plt.ylabel('Amplituda')
    plt.show()

