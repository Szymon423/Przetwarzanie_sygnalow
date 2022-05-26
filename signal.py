from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from datetime import datetime


class Signal:
    """klasa odpowiedzialana za przetwarzanie sygnału"""

    def __init__(self):
        """inicjacja obiektu klasy Signal"""

        # zmienna przechowująca adres do pliku z sygnałem
        self.file_adress = "Samples\\sin_1kHz.wav"

        # zmienna przechowująca sygnał - 2 kanały
        self.signal = np.zeros((2, 2))

        # zmienna przechowująca częstotliwość odtwarzania sygnału [Hz]
        self.samplerate = 8000

        # zmienna przechowująca długość sygnału
        self.signal_length = 0

        # zmienna odpowiedzialana za wybór kanału do obliczeń - domyślnie lewy
        self.canal = "left"

        # zmienna przechowująca wartość największej absolutnej wartości w sygnale - w celach porónawczych
        self.max_abs_val = 0

        # zmienna przechowująca sygnał zdecymowany
        self.decimated_signal = np.zeros(2)

    def define_path(self, path):
        """metoda odpowiedzialana za określenie adresu"""

        # przypisanie nowego adresu do ścieżki z klasy
        self.file_adress = path

    def load_singal(self):
        """metoda odpowiedzialna za wczytanie sygnału i określenie jego parametrów"""

        try:
            # wczytanie sygnału
            self.samplerate, self.signal = wavfile.read(self.file_adress)

            # wybór kanału do dalszych obliczeń
            if self.canal == "left":
                # dane są zapisane w postaci macierzy nx2 (zero to lewy kanał, 2 to prawy)
                self.signal = self.signal[:, 0]
            else:
                self.signal = self.signal[:, 1]

            # określenie długości sygnału
            self.signal_length = len(self.signal)

            # określenie największej wartości w sygnale
            self.max_abs_val = np.amax(np.abs(self.signal))

        except FileNotFoundError:
            print("Wrong file name..")
        else:
            print("Signal loaded successfully!")
            print("Signal length:", self.signal_length)
            print("Signal samplerate:", self.samplerate, "Hz")

    def normalise_signal(self):
        """metoda odpowiedzialana za normalizację sygnału -
           zamiana jego zakresu wartości do zakresu [0-1]
           realizowane to jest przez podzielenie każego elementu
           sygnału przez największą wartość w sygnale"""
        self.signal = self.signal / self.max_abs_val

        # print logu
        print("Signal normalised")

    def calculate_fft(self, _signal, _samplerate):
        """metoda służąca do obliczania fft z sygnału podanego jako argument """

        # obliczenie wartości surowej fft
        _raw_fft = fft(_signal)

        # skrócenie zakresu częstotilwości do połowy - usunięcie odbicia
        signal_fft_y = _raw_fft[0: len(_raw_fft) // 2]

        # dopasowanie częstotliwości do próbek w fft
        _values = np.array(range(len(_signal) // 2))
        signal_fft_x = _values / len(_signal) * _samplerate

        # print logu
        print("FFT computed")

        return signal_fft_x, signal_fft_y

    def decimate_signal(self, grade):
        """metoda odpowiedzialana za decymację sygnału w dziedzinie czasu
           wykorzystywana jest co grade-a próbka z sygnału początkowego"""

        _arr = []
        # zapis co 4 elementu
        for i in range(self.signal_length // grade):
            _arr.append(self.signal[grade * i])

        # przypisanie do docelowej tablicy
        self.decimated_signal = np.array(_arr)

        # print logu
        print("decimation computed")

    def periodogram(self, _signal, _samplerate):
        """metoda odpowiedzialana za przedstawienie periodogramu"""

        fx, pxx = signal.periodogram(_signal, _samplerate, 'hamming', 2048, scaling='density')
        return fx, pxx

        # print logu
        print("periodogram computed")