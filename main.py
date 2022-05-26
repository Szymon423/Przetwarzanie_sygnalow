from signal import Signal

# definicja obiektu sound
sound = Signal()

# określenie ścieżki dostępu
sound.define_path("Samples\\sin_1kHz.wav")

# wczytanie sygnału do obiektu
sound.load_singal()

# normalizacja sygnału
sound.normalise_signal()
print(sound.signal)
