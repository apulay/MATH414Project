import librosa
import pywt
import numpy as np

def featureExtraction(filepath):
    #y is the audio time series, sr is the sampling rate
    y, sr = librosa.load(filepath, sr=16000, duration=5.0)
    #Set sampling rate to 16kHz, as human speech is normally not understood below ~8kHz,
    #With Nyquist, the 16kHz captures the frequencies up to 8 kHz

    #Compress signal into meaningful information

    rms = librosa.feature.rms(y=y)[0]

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    #Measure of how many times the signal crosses the zero amplitude line, which can indicate the presence of noise

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    coeffs = pywt.wavedec(y, 'db4', level=4)

    #Calculating Means
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    mfcc_mean = np.mean(mfcc, axis=1)

    wavelet_energy = []
    for c in coeffs:
        wavelet_energy.append(np.sum(c**2))

    features = [rms_mean, rms_std, zcr_mean, zcr_std]
    features.extend(mfcc_mean)
    features.extend(wavelet_energy)

    return np.array(features)