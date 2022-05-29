from signal_processing import SignalProcessing
import matplotlib.pyplot as plt
import numpy as np

# definicja obiektu sound
sound = SignalProcessing()

# określenie ścieżki dostępu
sound.define_path("Samples\\example2.wav")

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
filtered_fft = sound.filter_signal(raw_fft, sound.samplerate, 2000, 5000)

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
plt.xlim([0,1000])
plt.ylim([-0.03, 0.03])
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.title('Przebiegi czasowe przed i po filtracji')
plt.legend(["oryginal", "filtrated"])
plt.show()
