from signal import Signal
import matplotlib.pyplot as plt
import numpy as np

# definicja obiektu sound
sound = Signal()

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

x = np.linspace(0,sound.signal_length, sound.signal_length)
x1 = np.linspace(0,sound.signal_length//decimate_grade, sound.signal_length//decimate_grade)
x1 = x1*decimate_grade

# przebieg czasowy
plt.figure(1)
plt.plot(x,sound.signal)
plt.plot(x1,sound.decimated_signal)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.title('Przebieg czasowy')
plt.legend(["oryginal", "decimated"])

# obliczenie fft dla wczytanego sygnału
fft_x_d, fft_y_d, _ = sound.calculate_fft(sound.decimated_signal, sound.samplerate // decimate_grade)

# wyświetlanie fft dla wczytanego sygnału oraz dla sygnału po decymacji
plt.figure(2)
plt.semilogy(fft_x, abs(fft_y), 'r')
plt.semilogy(fft_x_d, abs(fft_y_d), 'b')
#plt.xlim([0, 5000])
plt.xlabel('f [Hz]')
plt.ylabel('Amplituda')
plt.title('Widmo amplitudowe')
plt.legend(["oryginal", "decimated"])

# obliczenie periodogramu dla sygnału początkowego oraz zdecymowanego
fx, pxx = sound.periodogram(sound.signal, sound.samplerate)
fx_d, pxx_d = sound.periodogram(sound.decimated_signal, sound.samplerate // decimate_grade)
plt.figure(3)
plt.semilogy(fx, pxx)
plt.semilogy(fx_d, pxx_d)
plt.xlabel('f [Hz]')
plt.xlim([0, 5000])
plt.ylabel('Widmowa gęstość mocy')
plt.title('Periodogram przetwarzanego sygnału')
plt.legend(["oryginal", "decimated"])
plt.show()

# zapis do CSV danych z sygnału nie przetworzonego
sound.save_as_CSV(sound.signal, "time_domain")

# zapis do CSV danych z sygnału nie przetworzonego
sound.save_as_CSV(raw_fft[:len(raw_fft)//2], "freq_domain")