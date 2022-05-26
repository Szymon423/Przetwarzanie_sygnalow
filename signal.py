from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


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

        # zmienna przechowująca dane pochodzące z obliczeń fft
        self.signal_fft = np.zeros(2)

        # zmienna odpowiedzialana za wybór kanału do obliczeń - domyślnie lewy
        self.canal = "left"

        # zmienna przechowująca wartość największej absolutnej wartości w sygnale - w celach porónawczych
        self.max_abs_val = 0

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
           zamiana jego zakresu wartości do zakresu [0-1]"""
        self.signal = self.signal / self.max_abs_val

    def calculate_fft(self):
        """metoda służąca do obliczania ttf z sygnału dla """
        # self.signal_fft =