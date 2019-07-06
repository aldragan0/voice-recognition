import os
from pathlib import Path

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from librosa.display import specshow

from datasets import get_data, write_to_file_labels
from extract_features import get_audio_from_intervals
from file_io import load_metric, labels_from_file
from pitch_utils import smooth, get_pitch_magnitude
from utils import get_count

MFCC_FEATURES = 13
WORD_LENGTH = 35
SAMPLE_RATE = 44100


def audio_transform_plot(path):
    audio_time_series, _ = librosa.load(path, mono=True)

    blue_patch = mpatches.Patch(color='blue', label='Audio')
    orange_patch = mpatches.Patch(color='orange', label='Percussion')
    green_patch = mpatches.Patch(color='green', label='Harmonics')

    plt.figure(figsize=(12, 5))
    plt.subplot(4, 2, 1)
    plt.legend(handles=[blue_patch, orange_patch, green_patch])
    y_harmonic = librosa.effects.harmonic(audio_time_series)
    y_percussion = librosa.effects.percussive(audio_time_series)

    plt.plot(audio_time_series)
    plt.plot(y_percussion)
    plt.plot(y_harmonic)

    plt.subplot(4, 2, 3)
    intervals = librosa.effects.split(audio_time_series, top_db=18)
    audio_time_series = get_audio_from_intervals(audio_time_series, intervals)
    plt.plot(audio_time_series)


    plt.subplot(4, 2, 5)
    audio_time_series, _ = librosa.effects.trim(audio_time_series, top_db=10)
    y_harmonic = librosa.effects.harmonic(audio_time_series)
    plt.plot(audio_time_series)
    plt.plot(y_harmonic)


def mfcc_features_plot(path):
    audio_time_series, sampling_rate = librosa.load(path, mono=True)

    mfcc_features = librosa.feature.mfcc(y=audio_time_series, sr=sampling_rate, n_mfcc=MFCC_FEATURES)
    if mfcc_features.shape[1] > WORD_LENGTH:
        mfcc_features = mfcc_features[:, 0:WORD_LENGTH]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, WORD_LENGTH - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)  # extend matrix with values of 0

    plt.subplot(4, 2, 2)
    librosa.display.specshow(mfcc_features)
    plt.colorbar()
    plt.title("MFCC")

    mfcc_delta = librosa.feature.delta(mfcc_features)
    plt.subplot(4, 2, 4)
    plt.title(r'MFCC-$\Delta$')
    librosa.display.specshow(mfcc_delta)
    plt.colorbar()

    mfcc_delta_2 = librosa.feature.delta(mfcc_features, order=2)
    plt.subplot(4, 2, 6)
    plt.title(r'MFCC-$\Delta^2$')
    librosa.display.specshow(mfcc_delta_2, x_axis='time')
    plt.colorbar()


def plot_figures(pitches, magnitudes):

    plt.subplot(4, 2, 7)
    pitches = smooth(pitches, window_len=10)
    plt.plot(pitches)
    plt.title("Pitches")

    plt.subplot(4, 2, 8)
    plt.plot(magnitudes)
    plt.title("Magnitudes")

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


def other_feature_plot(path):
    audio_time_series, sampling_rate = librosa.load(path, mono=True, sr=SAMPLE_RATE)
    intervals = librosa.effects.split(audio_time_series, top_db=18)
    audio_time_series = get_audio_from_intervals(audio_time_series, intervals)

    audio_time_series, _ = librosa.effects.trim(audio_time_series, top_db=10)

    stft = np.abs(librosa.stft(audio_time_series))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sampling_rate)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sampling_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_time_series),
                                      sr=sampling_rate)
    center = librosa.feature.spectral_centroid(audio_time_series, sr=sampling_rate, hop_length=128)

    plt.figure(figsize=(12, 4))
    plt.subplot(5, 1, 1)
    plt.plot(audio_time_series)
    plt.title("Audio data")

    plt.subplot(5, 1, 2)
    print(chroma.shape)
    librosa.display.specshow(chroma)
    plt.colorbar()
    plt.title("Chromagram")

    plt.subplot(5, 1, 3)
    print(contrast.shape)
    librosa.display.specshow(contrast)
    plt.colorbar()
    plt.title("Spectral contrast")

    plt.subplot(5, 1, 4)
    print(tonnetz.shape)
    librosa.display.specshow(tonnetz)
    plt.colorbar()
    plt.title("Tonnetz features")

    plt.subplot(5, 1, 5)
    print(center.shape)
    librosa.display.specshow(center, x_axis='time')
    plt.colorbar()
    plt.title("Spectral center")

    plt.show()


def dataset_plot(data_file):
    labels = labels_from_file(data_file)
    counts = get_count(labels)
    counts_x = list(counts.keys())
    counts_y = list(counts.values())
    plt.bar([i * 2 for i in range(len(counts_x))], counts_y, tick_label=counts_x)
    plt.show()


def model_performance_plot(model_data_path, num_epochs, metrics):

    model_name = model_data_path.split('/')[1]
    if model_data_path[-1] != '/':
        model_data_path += '/'
    model_data_path += model_name + "_" + str(num_epochs)

    plt.figure(figsize=(12, 4))

    for i in range(len(metrics)):

        history = load_metric(model_data_path + "_" + metrics[i])

        plt.subplot(len(metrics), 1, i + 1)

        min_el = min(np.minimum(history[:, 0], history[:, 1]))
        max_el = max(np.maximum(history[:, 0], history[:, 1]))
        plt.yticks(np.arange(min_el, max_el + 1, (max_el - min_el) / 5.0 ))

        plt.plot(history)
        plt.title(model_data_path + "-" + metrics[i])
        plt.xlabel("epoch")
        plt.ylabel(metrics[i])
        plt.legend(["train", "test"], loc="upper left")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # the process of transforming audio before it gets converted to mfcc

    # Dataset plot
    # dataset_plot("data/gender_out")
    # dataset_plot("data/age_out")
    
    # Audio features plot
    data_path = "audio/"

    file = os.listdir(data_path)[0]
    file = Path(data_path + file)
    duration = librosa.core.get_duration(filename=str(file))

    audio_transform_plot(file)
    mfcc_features_plot(file)
    audio_data, pitch_values, magnitude_values = get_pitch_magnitude(file, SAMPLE_RATE)
    plot_figures(pitch_values, magnitude_values)
    # other_feature_plot(file)

    # Model Performance plot
    # model_performance_plot("model/lstm_gender_41", 10, ['acc', 'loss', 'precision'])
    # model_performance_plot("model/lstm_age_41", 30, ['acc', 'loss', 'precision'])
    # model_performance_plot("model/lstm_lang_41", 20, ['acc', 'loss', 'precision'])
