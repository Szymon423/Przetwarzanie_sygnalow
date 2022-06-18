from signal_processing import SignalProcessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter
from scipy import signal
from scipy.io.wavfile import write

# definicja obiektu sound
sound = SignalProcessing()

# określenie ścieżki dostępu
sound.define_path("Samples\\panTadeusz_noise.wav")

# wczytanie sygnału do obiektu
sound.load_singal()

# normalizacja sygnału
sound.normalise_signal()

# plt.figure(1)
# plt.plot(sound.signal)
# plt.show()

# obliczenie fft dla wczytanego sygnału
fft_x, fft_y, raw_fft = sound.calculate_fft(sound.signal, sound.samplerate)

# filtracja sygnału oryginalnego
filtered_fft = sound.filter_signal(raw_fft, sound.samplerate, 0, 2000)

# obliczenie transformatyodwrotnej
filtered_sound = sound.calculate_invers_fft(filtered_fft)

filtered_sound = filtered_sound * sound.max_abs_val

plt.figure(1)
plt.plot(filtered_fft)
plt.show()

plt.figure(2)
plt.plot(sound.signal)
plt.plot(filtered_sound)
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.title('Przebiegi czasowe przed i po filtracji')
plt.legend(["oryginal", "filtrated"])
plt.show()

write("save.wav", sound.samplerate, filtered_sound.astype(np.int16))