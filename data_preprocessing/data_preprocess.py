import pyedflib
import numpy as np
from scipy.signal import resample
import warnings

def check_arrays_length(arr1, arr2, arr3):
    if len(arr1) != len(arr2) or len(arr1) != len(arr3):
        raise ValueError("Signal, Artefact, and Hypnogram channels must have the same length.")

def read_edf_file(file_name, channel_index):
    edf_file = pyedflib.EdfReader(file_name)
    signal_labels = edf_file.getSignalLabels()
    channel_data = edf_file.readSignal(channel_index)
    edf_file._close()
    del edf_file
    return channel_data, signal_labels[channel_index]

def edf_info(file_name):
    edf_file = pyedflib.EdfReader(file_name)
    signal_labels = edf_file.getSignalLabels()
    eeg_sampling_rate = edf_file.getSignalHeader(0)['sample_rate']
    print(signal_labels)
    print(f"Sample Rate: {eeg_sampling_rate / 10} Hz")
    print('Length of the signal in seconds:', edf_file.file_duration)
    signal_headers = edf_file.getSignalHeaders()
    for i, header in enumerate(signal_headers):
        print('Signal', i+1, 'label:', header['label'])
        print('Sampling rate:', header['sample_rate'])
    edf_file._close()
    del edf_file

def read_channel(file_name, channel_name):
    edf_file = pyedflib.EdfReader(file_name)
    signal_labels = edf_file.getSignalLabels()
    channel_index = signal_labels.index(channel_name)
    channel_data = edf_file.readSignal(channel_index)
    edf_file._close()
    del edf_file
    return channel_data

def epoch_signal(signal_data, sampling_rate, epoch_duration):
    num_samples_per_epoch = epoch_duration * sampling_rate
    num_epochs = len(signal_data) // num_samples_per_epoch
    signal_data_epochs = np.array_split(signal_data[:num_epochs * num_samples_per_epoch], num_epochs)
    return signal_data_epochs

def check_channels_lengths(file_name):
    edf_file = pyedflib.EdfReader(file_name)
    signal_labels = edf_file.getSignalLabels()
    for index, channel in enumerate(signal_labels):
        print(f"channel: {channel} length: {len(edf_file.readSignal(index))}")

def merge_stage_signal(stage_signal):
    full_stage_signal = []
    for i in range(len(stage_signal) - 1):
        full_stage_signal += stage_signal[i]
    return full_stage_signal

def resample_signal(signal, original_rate, target_rate):
    resampling_ratio = target_rate / original_rate
    new_length = int(len(signal) * resampling_ratio)
    original_time = np.arange(0, len(signal)) / original_rate
    with warnings.catch_warnings():  # Suppress resample's warnings
        warnings.simplefilter("ignore")
        resampled_signal = resample(signal, new_length)
    return resampled_signal

def clean_data_from_art_big5(signal, art, hypno):
    check_arrays_length(signal, art, hypno)


    clean_signal = []
    clean_art = []
    clean_hypno = []

    for index, value in enumerate(art):
        if value <= 5:

            clean_signal.append(signal[index])
            clean_art.append(value)
            clean_hypno.append(hypno[index])

    clean_signal = np.array(clean_signal, dtype=float)
    clean_art = np.array(clean_art, dtype=int)
    clean_hypno = np.array(clean_hypno, dtype=int)

    return clean_signal, clean_hypno