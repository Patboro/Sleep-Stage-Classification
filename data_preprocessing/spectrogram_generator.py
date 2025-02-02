import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, detrend
import os


def calculate_epochs(signal, fs, epoch_length):

    samples_per_epoch = epoch_length * fs
    num_epochs = len(signal) // samples_per_epoch
    eeg_epochs = np.reshape(signal[:num_epochs * samples_per_epoch], (num_epochs, samples_per_epoch))
    return eeg_epochs


def calculate_spectrogram(epoch, fs, NFFT, noverlap, desired_freq_range):

    epoch_detrended = detrend(epoch, type='linear')
    freqs, times, Sxx = spectrogram(epoch_detrended, fs=fs, nperseg=NFFT, noverlap=noverlap, scaling='density')
    freq_range_indices = np.where((freqs >= desired_freq_range[0]) & (freqs <= desired_freq_range[1]))
    cropped_Sxx = Sxx[freq_range_indices]
    return freqs, times, cropped_Sxx, freq_range_indices


def plot_spectrogram(freqs, times, cropped_Sxx, freq_range_indices, label, i, save_folder):

    plt.figure()
    plt.pcolormesh(times, freqs[freq_range_indices], np.log10(cropped_Sxx), shading='gouraud', cmap='jet')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    img_path = f'Stage_{label}_Epoch_{i + 1}'
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f"{save_folder}/{img_path}", bbox_inches='tight', pad_inches=0)
    plt.close()


def eeg_spectrogram(signal, fs, label, save_folder):

    desired_freq_range = (0, 50)
    epoch_length = 30
    NFFT = 1024
    noverlap = 512


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    eeg_epochs = calculate_epochs(signal, fs, epoch_length)

    for i, epoch in enumerate(eeg_epochs):
        freqs, times, cropped_Sxx, freq_range_indices = calculate_spectrogram(epoch, fs, NFFT, noverlap, desired_freq_range)
        plot_spectrogram(freqs, times, cropped_Sxx, freq_range_indices, label, i, save_folder)

