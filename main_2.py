from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import scipy.fft


#filename = input('Podaj ścieżkę do pliku .wav: ')
filename = "sin_1kHz"
print("time")
# try:
# 	f = open(filename,'r')
# 	f.close()
# except FileNotFoundError :
# 	print('Błąd wczytania pliku.')
# 	quit()

samplerate, data = wavfile.read("Samples\\" + filename + ".wav")

print(data[:,0])
print(data[:,1])
print('Częstotliwość próbkowania [Hz]: ', samplerate)


# czas trwania sygnału
time = data.shape[0] / samplerate 
x = np.linspace(0,time,data.shape[0])

#print(data.shape)
print('Wymiar przetwarzanego sygnału (ilość próbek): ', data.shape[0])
print('Czas trwania sygnału [s]: ', time)

# przebieg czasowy
# plt.figure(1)
# plt.plot(x,data[:,0])
# plt.xlabel('Czas [s]')
# plt.ylabel('Amplituda')
# plt.title('Przebieg czasowy')

# widmo amplitudowe - transformata FFT
b = np.array([(ele/2**13.)*2-1 for ele in data])
c= fft(b[:,0])
d = len(c/2)
ff = np.array(range(data.shape[0]//2))
e = ff/data.shape[0]*samplerate
plt.figure(2)
plt.plot(e[10:],20*np.log10(abs(c[:(d)]))[10:data.shape[0]//2],'r') #skala w decybelach
#plt.ylim(60,120)
#plt.semilogy(e[10:],abs(c[:(d)])[10:data.shape[0]//2],'r') #skala logarytmiczna
#plt.ylim(10**2,10**5)
plt.xlabel('f [Hz]')
plt.ylabel('Amplituda [dB]')
plt.title('Widmo amplitudowe')

# decymacja
nn = data.size
q = 4  #ilość decymacji (stopień M)
sig_size = nn // q
# c - fft dla jednego kanału
s1 = c[:len(c) // q + 1]
s1[1:-1] /= q
f1 = np.fft.rfftfreq(sig_size, 1/samplerate)
plt.figure(3)
plt.plot(f1,20*np.log10((abs(s1))/(samplerate/2)))
plt.xlabel('f [Hz]')
plt.ylabel('Amplituda [dB]')
plt.title('Widmo sygnału po decymacji')

x1 = np.real(np.fft.irfft(s1))
fig, ax = plt.subplots(2)
ax[0].plot(data[:, 0][10:10000], label="fs=48 kHz") #zakres próbek dla sygnału przed
ax[1].plot(x1[10:10000], label="fs=4,8 kHz")    #zakres próbek dla sygnału po decymacji
for a in ax:
    a.set_xlabel('Numer próbki')
    a.set_ylabel('Amplituda')
    a.legend(loc='upper right')
ax[0].set_title('Sygnał przed i po decymacji FFT')

#periodogram - rozkład mocy widmowej sygnału na jednostkę częstotliwości
fx, Pxx = signal.periodogram(data[:,0],samplerate,'hamming', 2048, scaling ='density')
plt.figure(5)
plt.semilogy(fx, Pxx)
plt.xlabel('f [Hz]')
plt.ylabel('Widmowa gęstość mocy')
plt.title('Periodogram przetwarzanego sygnału')
plt.show()


# odwrotna transformata Fouriera - IFFT
# cc = scipy.fft.ifft(c)
# plt.figure(6)
# plt.plot(x,cc.real, 'b')
# plt.plot(x,cc.imag, 'y')
# plt.xlabel('Czas [s]')
# plt.ylabel('Amplituda')
# plt.title('Przebieg czasowy po zastosowaniu IFFT')
# plt.legend(['real','imaginary'])
# plt.show()

