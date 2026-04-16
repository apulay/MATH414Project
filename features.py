import librosa
import numpy as np
#y is the audio time series, sr is the sampling rate
y, sr = librosa.load('data/AI/AI1.mp3', sr = 16000, duration = 5.0)
#Set sampling rate to 16kHz, as human speech is normally not understood below ~8kHz,
#With Nyquist, the 16kHz captures the frequencies up to 8 kHz

#Compress signal into meaningful information
rms = librosa.feature.rms(y=y)
#Split up audio into short frames, and find energy/loudness per frame