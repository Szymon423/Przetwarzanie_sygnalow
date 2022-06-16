from signal_processing import SignalProcessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import scipy
from scipy import signal
from scipy.signal import butter
from scipy.signal import filtfilt


# definicja obiektu sound
sound = SignalProcessing()

# określenie ścieżki dostępu
sound.define_path("Samples\\panTadeusz_ijo_ijo.wav")
# sound.define_path("noise+łeło_łeło.wav")

# wczytanie sygnału do obiektu
sound.load_singal()

# normalizacja sygnału
sound.normalise_signal()

# obliczenie fft dla wczytanego sygnału
fft_x, fft_y, raw_fft = sound.calculate_fft(sound.signal, sound.samplerate)

# określenie stopnia decymacji
decimate_grade = 5

# wykonanie decymacji
sound.decimate_signal(decimate_grade)

x = np.linspace(0, sound.signal_length, sound.signal_length)
x1 = np.linspace(0, sound.signal_length // decimate_grade, sound.signal_length // decimate_grade)
x1 = x1 * decimate_grade

# # przebieg czasowy
# plt.figure(1)
# plt.plot(x,sound.signal)
# #plt.plot(x1,sound.decimated_signal)
# plt.xlim([0,1000])
# plt.ylim([-0.03, 0.03])
# plt.xlabel('Numer próbki')
# plt.ylabel('Amplituda')
# plt.title('Przebieg sygnału')
#plt.legend(["oryginal", "decimated"])
#
# obliczenie fft dla wczytanego sygnału
fft_x_d, fft_y_d, _ = sound.calculate_fft(sound.decimated_signal, sound.samplerate // decimate_grade)
#
# # wyświetlanie fft dla wczytanego sygnału oraz dla sygnału po decymacji
# plt.figure(2)
# plt.semilogy(fft_x, abs(fft_y), 'r')
# plt.semilogy(fft_x_d, abs(fft_y_d), 'b')
# #plt.xlim([0, 5000])
# plt.xlabel('f [Hz]')
# plt.ylabel('Amplituda')
# plt.title('Widma amplitudowe')
# plt.legend(["oryginal", "decimated"])
#
# # obliczenie periodogramu dla sygnału początkowego oraz zdecymowanego
# fx, pxx = sound.periodogram(sound.signal, sound.samplerate)
# fx_d, pxx_d = sound.periodogram(sound.decimated_signal, sound.samplerate // decimate_grade)
# plt.figure(3)
# plt.semilogy(fx, pxx)
# plt.semilogy(fx_d, pxx_d)
# plt.xlabel('f [Hz]')
# #plt.xlim([0, 5000])
# plt.ylabel('Widmowa gęstość mocy')
# plt.title('Periodogramy przetwarzanych sygnałów')
# plt.legend(["oryginal", "decimated"])
# plt.show()
#
#zapis do CSV danych z sygnału nieprzetworzonego
sound.save_as_CSV(sound.signal, "time_domain")

#zapis do CSV danych z sygnału nieprzetworzonego
sound.save_as_CSV(raw_fft[:len(raw_fft)//2], "freq_domain")

# filtracja sygnału oryginalnego
filtered_fft = sound.filter_signal(raw_fft, sound.samplerate, 6000, 25000)
# filtered_fft = sound.filter_signal(filtered_fft, sound.samplerate, 6000, 25000)

# obliczenie transformatyodwrotnej
filtered_sound = sound.calculate_invers_fft(filtered_fft)

# obliczenie fft na podstawie nowego sygnału - po filtracji
x_new, y_new, _ = sound.calculate_fft(filtered_sound, sound.samplerate)



plt.figure(4)
plt.semilogy(fft_x, abs(fft_y), 'b')
plt.semilogy(x_new, abs(y_new), 'r')
plt.ylim([10**(-2), 10**5])
plt.xlabel('f [Hz]')
plt.ylabel('Amplituda')
plt.title('Porównanie fft przed i po filtracji')
plt.legend(["oryginal", "filtrated"])

# przebieg czasowy przed i po filtracji
plt.figure(5)
plt.plot(sound.signal)
plt.plot(filtered_sound)
plt.xlim([0, 1000])
plt.ylim([-0.03, 0.03])
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.title('Przebiegi czasowe przed i po filtracji')
plt.legend(["oryginal", "filtrated"])


fft_compr = []
fft_with_zeros = raw_fft.copy()
treshold = 10 ** 1.8

for index, sample in enumerate(abs(raw_fft)):
    if sample < treshold:
        fft_with_zeros[index] = 0
    else:
        fft_compr.append(raw_fft[index])

fft_compr = np.array(fft_compr)

dupa = sound.calculate_invers_fft(fft_with_zeros)
print(np.amax(abs(dupa)))
print(np.amax(abs(sound.signal)))

# przebieg czasowy przed i po filtracji
plt.figure(6)
plt.plot(sound.signal)
plt.plot(dupa)
plt.xlim([10000, 11000])
# plt.ylim([-0.03, 0.03])
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.title('Przebiegi czasowe przed i po kompresji')
plt.legend(["oryginal", "compressed"])

compres_x, compres_y, _ = sound.calculate_fft(dupa, sound.samplerate)

plt.figure(7)
plt.semilogy(fft_x, abs(fft_y), 'b')
plt.semilogy(compres_x, abs(compres_y), 'r')
# plt.ylim([10**(-2), 10**5])
plt.xlabel('f [Hz]')
plt.ylabel('Amplituda')
plt.title('Porównanie fft przed i po kompresji')
plt.legend(["oryginal", "compressed"])
plt.show()

print("dupa len:", len(dupa))
print("sound len:", sound.signal_length)

sound.save_as_CSV(fft_compr, "1.8")

filtered_sound = filtered_sound * sound.max_abs_val

fs = 44100/2
lowcut = 20
highcut = 50

nyq = 0.5 * fs
low = 2000
high = 5000

order = 2

b, a = butter(3, 0.05)
# 0.15 dla magic
# 0.05 dla ijoijo

y = scipy.signal.filtfilt(b, a, sound.signal)
y = y * sound.max_abs_val
write("save.wav", sound.samplerate, y.astype(np.int16))

plt.figure(8)
plt.plot(sound.signal * sound.max_abs_val)
plt.plot(y)
plt.xlim([10000, 15000])
# plt.ylim([-0.03, 0.03])
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.title('Przebiegi czasowe przed i po kompresji')
plt.legend(["oryginal", "compressed"])
plt.show()
