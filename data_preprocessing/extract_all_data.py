
from data_preprocess import read_channel, resample_signal, clean_data_from_art_big5
from epoch_data import split_signal_by_sleep_stage
from spectrogram_generator import eeg_spectrogram
import os

def read_channels(file):
    fp1_signal = read_channel(file, 'Fp1')
    artff_fp1 = read_channel(file, 'ArtiFFT Fp1-A2')
    hypno = read_channel(file, 'Hypnogram')
    return fp1_signal, artff_fp1, hypno

def upsample_channels(art, hypno):
    art_upsampled = [int(x) for x in resample_signal(art, 1, 256)]
    hypno_upsampled = [int(x) for x in resample_signal(hypno, 0.1, 256)]
    return art_upsampled, hypno_upsampled

if __name__ == "__main__":
    path = "ResNet_EEG_Sleep_Classification/files/"
    files = ["BB0008_SP2", "BB0008_SP3", "BB0009_SP2", "BB0008_SP3"]
    save_path = "D:/4Bachelor/ResNet_EEG_Sleep_Classification/Spectograms/"
    save_folders = ["B8_SP2", "B8_SP3", "B9_SP2", "B9_SP3"]

    for file, save_folder in zip(files, save_folders):
        print(f"Working on file...{file}")
        signal = read_channels(f"{path}{file}.edf")
        art_upsampled, hypno_upsampled = upsample_channels(signal[1], signal[2])
        c_signal, c_hypno = clean_data_from_art_big5(signal[0], art_upsampled, hypno_upsampled)

        sleep_stages = {
            '0': 'Wake',
            '1': 'N1',
            '2': 'N2',
            '3': 'N3',
            '4': 'REM'
        }

        for stage_label, stage_name in sleep_stages.items():
            stage_signal = split_signal_by_sleep_stage(c_signal, c_hypno, int(stage_label))
            stage_save_folder = os.path.join(save_path, save_folder, stage_name)
            eeg_spectrogram(stage_signal, fs=256, label=stage_label, save_folder=stage_save_folder)

    print("Processing complete.")
