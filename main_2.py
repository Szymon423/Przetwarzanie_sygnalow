from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import scipy.fft
import math

#filename = input('Podaj ścieżkę do pliku .wav: ')
filename = "example"
print("time")
# try:
# 	f = open(filename,'r')
# 	f.close()
# except FileNotFoundError :
# 	print('Błąd wczytania pliku.')
# 	quit()

samplerate, data = wavfile.read("Samples\\" + filename + ".wav")

# print(data[:,0])
# print(data[:,1])
print('Częstotliwość próbkowania [Hz]: ', samplerate)

# czas trwania sygnału
time = data.shape[0] / samplerate 
x = np.linspace(0,time,data.shape[0])
xx = data[:,0]

print('Wymiar przetwarzanego sygnału (ilość próbek): ', data.shape[0])
print('Czas trwania sygnału [s]: ', time)

a = (2**math.ceil(np.log2(len(x)))-len(x))      #ceil lub floor - obliczenie ilości próbek potrzebnych do osiągnięcia liczby próbek, która jest potęgą 2
syg_2 = np.zeros(len(x)+a)         # wypełnianie brakujących próbek zerami
print("Ilość próbek zerowych dodana do sygnału: ", a)
print("Długość sygnału uzupełnionego zerami: ", len(syg_2))
for i in range(0,len(x)):       # dla ceil len(x), dla floor len(syg_2)
    syg_2[i] = xx[i]

### Przebieg czasowy sygnału oryginalnego i uzupełnionego zerami
# plt.figure(1)
# plt.plot(syg_2)
# plt.plot(xx)
# plt.xlabel('Czas [s]')
# plt.ylabel('Amplituda')
# plt.title('Przebieg czasowy')

'''Liczenie DFT z definicji - brak pamięci'''
# def DFT(x):
#     """Function to calculate the discrete Fourier Transform of a 1D real-valued signal x"""
#     N = len(x)
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     e = np.exp(-2j * np.pi * k * n / N)
#     X = np.dot(e, x)
#     return X
# X = DFT(x)
# # calculate the frequency
# N = len(X)
# n = np.arange(N)
# T = N / samplerate
# freq = n / T
#
# plt.figure(1)
# plt.stem(freq, abs(X), 'b', \
#          markerfmt=" ", basefmt="-b")
# plt.xlabel('f [Hz]')
# plt.ylabel('Amplituda')
# plt.legend(["DFT definition"])
# plt.show()

def FFT(x):
    """ A recursive implementation of the 1D Cooley-Tukey FFT, the input should have a length of power of 2."""
    N = len(x)
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])        # ciąg parzysty
        X_odd = FFT(x[1::2])        # ciąg nieparzysty
        factor = \
            np.exp(-2j * np.pi * np.arange(N) / N)
        # łączenie ciągów parzystych i nieparzystych
        X = np.concatenate( \
            [X_even + factor[:int(N / 2)] * X_odd,
             X_even + factor[int(N / 2):] * X_odd])
        return X

X=FFT(syg_2)
# obliczenie częstotliwości
N = len(X)
n = np.arange(N)
T = N/samplerate
freq = n/T
n_half = N//2               # Bierzemy pod uwagę połowę widma
f_half = freq[:n_half]      # Częstotliwość dla połowy widma
X_half =X[:n_half]/n_half   # Normalizacja amplitudy

fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16,8))
ax1.semilogy(f_half, abs(X_half), 'b')
#plt.ylim(10**-4,10**5)
ax1.set_xlabel('f [Hz]')
ax1.set_ylabel('Amplituda')
ax1.set_title('Widmo amplitudowe sygnału po transformacie obliczonej z definicji')
ax1.legend(["FFT definition"])

# widmo amplitudowe - przy użyciu transformaty FFT
b = np.array([(ele/2**13.)*2-1 for ele in data])
c = fft(b[:,0])
d = len(c/2)
ff = np.array(range(data.shape[0]//2))
e = ff/data.shape[0]*samplerate
# plt.plot(e[10:],20*np.log10(abs(c[:(d)]))[10:data.shape[0]//2],'r') #skala w decybelach
ax2.semilogy(e[10:],abs(c[:(d)])[10:data.shape[0]//2],'r') #skala logarytmiczna
#plt.ylim(10**2,10**5)
ax2.set_xlabel('f [Hz]')
ax2.set_ylabel('Amplituda')
ax2.set_title('Widmo amplitudowe sygnału po FFT')
ax2.legend(["FFT"])

# Sprawdzenie wyników transformat
print('Własna funkcja FFT: ')
print(FFT(syg_2))
print('numpy.fft.fft: ')
print(fft(xx))

plt.show()

