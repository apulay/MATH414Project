import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pywt

def _ensure_outdir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)


def plot_waveform(filepath, outdir="data/visualizations", sr=16000, save=True, show=False):
    """Save or show the waveform for a file."""
    _ensure_outdir(outdir)
    y, sr = librosa.load(filepath, sr=sr)
    times = np.arange(len(y)) / float(sr)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, y, color="black", linewidth=0.7)
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    basename = os.path.splitext(os.path.basename(filepath))[0]
    outpath = os.path.join(outdir, f"{basename}_waveform.png")
    if save:
        fig.savefig(outpath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return outpath


def plot_dwt(filepath, outdir="data/visualizations", sr=16000, wavelet="db4", level=4, save=True, show=False):
    """Save or show a discrete wavelet transform visualization for a file."""
    _ensure_outdir(outdir)
    y, sr = librosa.load(filepath, sr=sr)
    coeffs = pywt.wavedec(y, wavelet, level=level)

    fig, axes = plt.subplots(len(coeffs), 1, figsize=(12, 2.2 * len(coeffs)), sharex=False)
    if len(coeffs) == 1:
        axes = [axes]

    for index, coeff in enumerate(coeffs):
        label = "Approximation" if index == 0 else f"Detail L{level - index + 1}"
        axes[index].plot(coeff, color="steelblue", linewidth=0.7)
        axes[index].set_ylabel(label)
        axes[index].grid(alpha=0.2)

    axes[-1].set_xlabel("Coefficient index")
    fig.suptitle("Discrete Wavelet Transform")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    basename = os.path.splitext(os.path.basename(filepath))[0]
    outpath = os.path.join(outdir, f"{basename}_dwt.png")
    if save:
        fig.savefig(outpath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return outpath
