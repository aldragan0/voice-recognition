from threading import Thread

import librosa
import numpy as np
from scipy.signal import lfilter

import file_io as io
from pitch_utils import get_pitch_magnitude, smooth

NUM_TIMESTAMPS = 35
BATCH_SIZE = 128
SAMPLE_RATE = 44100
MFCC_FEATURES = 13

# 41
# mfcc_feature_count -- num of frequency bins in which to split the audio file


def get_audio_from_intervals(audio_data, intervals):
    """
    concatenate audio data from the given intervals
    :param audio_data: data to be used
    :param intervals: intervals to keep from audio data
    :return: data containing only the kept intervals
    """
    new_audio = []
    for start, end in intervals:
        new_audio.extend(audio_data[start: end])
    return np.asarray(new_audio)


def get_audio_features(path, extra_features=[""]):
    """
    currently supports extraction for the following features from a given audio file
    -  mfcc features
    -  delta mfcc features
    -  delta-delta mfcc features
    -  shifted delta coeff
    -  pitch estimate
    -  magnitude estimate
    example:
        features = get_audio_features("file_path.mp3", extra_features=["delta", "delta2"])
    :param extra_features: features to take into consideration
    :param path: filePath
    :return: mfcc_features: matrix with shape (num_features, NUM_TIMESTAMPS)
    """
    audio_time_series, pitch_vals, magnitude_vals = get_pitch_magnitude(path, SAMPLE_RATE)

    pitch_vals = smooth(pitch_vals, window_len=10)

    # process audio - remove silence
    intervals = librosa.effects.split(audio_time_series, top_db=18)
    audio_time_series = get_audio_from_intervals(audio_time_series, intervals)
    audio_time_series, _ = librosa.effects.trim(audio_time_series, top_db=10)

    mfcc_features = librosa.feature.mfcc(y=audio_time_series, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)

    if mfcc_features.shape[1] > NUM_TIMESTAMPS:
        mfcc_features = mfcc_features[:, 0:NUM_TIMESTAMPS]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, NUM_TIMESTAMPS - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)

    if pitch_vals.shape[0] > NUM_TIMESTAMPS:
        pitch_vals = pitch_vals[0:NUM_TIMESTAMPS]
    else:
        pitch_vals = np.pad(pitch_vals, ((0, 0), (0, NUM_TIMESTAMPS - pitch_vals.shape[0])),
                            mode='constant', constant_values=0)

    if magnitude_vals.shape[0] > NUM_TIMESTAMPS:
        magnitude_vals = magnitude_vals[0:NUM_TIMESTAMPS]
    else:
        magnitude_vals = np.pad(magnitude_vals, ((0, 0), (0, NUM_TIMESTAMPS - magnitude_vals.shape[0])),
                                mode='constant', constant_values=0)

    features = []
    features.extend(mfcc_features)

    if "delta" in extra_features:
        mfcc_delta = librosa.feature.delta(mfcc_features)
        features.extend(mfcc_delta)

    if "delta2" in extra_features:
        mfcc_delta_2 = librosa.feature.delta(mfcc_features, order=2)
        features.extend(mfcc_delta_2)

    if "sdc" in extra_features:
        sdc_coeff = shifted_delta_coefficients(mfcc_features)
        features.extend(sdc_coeff)

    if "pitch" in extra_features:
        features.append(pitch_vals)
        features.append(magnitude_vals)

    return np.asarray(features)


def shifted_delta_coefficients(mfcc_coef, d=1, p=3, k=7):
    """
    :param mfcc_coef: mfcc coefficients
    :param d: amount of shift for delta computation
    :param p: amount of shift for next frame whose deltas are to be computed
    :param k: no of frame whose deltas are to be stacked.
    :return: SDC coefficients
    reference code from: http://tiny.cc/q8d58y
    """
    total_frames = mfcc_coef.shape[1]
    padding = mfcc_coef[:p * (k - 1), :]

    mfcc_coef = np.hstack((mfcc_coef, padding)).T

    deltas = mfcc_to_delta(mfcc_coef, d).T
    sd_temp = []

    for i in range(k):
        temp = deltas[:, p * i + 1:]
        sd_temp.extend(temp[:, :total_frames])
    sd_temp = np.asarray(sd_temp)
    return sd_temp


def mfcc_to_delta(mfcc_coef, d):
    """
    compute delta coefficient
    :param mfcc_coef: mfcc coefficients where a row represents the feature
                      vector for a frame
    :param d: lag size for delta feature computation
    reference code from: http://tiny.cc/m6d58y
    """
    num_frames, num_coeff = mfcc_coef.shape
    vf = np.asarray([i for i in range(d, -1-d, -d)])
    vf = vf / sum(vf ** 2)
    ww = np.ones(d).astype(np.int)
    cx = np.vstack((mfcc_coef[ww, :], mfcc_coef))
    cx = np.vstack((cx, mfcc_coef[(num_frames * ww) - 1, :]))
    vx = np.reshape(lfilter(vf, 1, cx[:]), (num_frames + 2 * d, num_coeff))
    mask = np.ones(vx.shape, dtype=np.bool)
    mask[: d * 2, :] = False
    vx = np.reshape(vx[mask], (num_frames, num_coeff))
    return vx


def create_intervals(data_len, interval_no):
    """
    split the interval [0, data_len] into a list of equal intervals
    :param data_len: length of the data
    :param interval_no: number of intervals to be split into
    :return: list of intervals (start, end)
    """
    step = data_len // interval_no
    intervals = []
    for start in range(0, data_len - step + 1, step):
        end = start + step
        if data_len - (end + step) < 0:
            end = data_len
        intervals.append((start, end))
    return intervals


def write_to_file_audio_data(output_file, audio_list, start, end, features):
    """
    :param end: interval end
    :param start: interval start
    :param output_file: all extracted data from mp3 files
    :param audio_list: audio file paths
    :param features: features to extract
    """
    index = 0
    for el in range(start, end):
        if index % 100 == 0:
            print("[({}, {}) Completed: {:.2f}]".format(start, end, index / (end - start)))
        matrix = get_audio_features(audio_list[el], extra_features=features)
        io.write_to_file_features(output_file, matrix)
        index += 1


def get_features(prefix, input_files, features):
    """
    transform audio to mfcc features
    :param prefix:
    :param input_files: files to transform
    :param features: features to select from the audio files
    """
    no_threads = 6
    intervals = create_intervals(len(input_files), no_threads)
    files = [prefix + "input" + str(i + 1) for i in range(no_threads)]
    threads = []
    print(len(input_files))
    print(intervals)
    print(files)
    file_index = 0

    for start, end in intervals:

        thread = Thread(target=write_to_file_audio_data,
                        args=[files[file_index], input_files, start, end, features])
        thread.start()
        threads.append(thread)
        file_index += 1

    for thr in threads:
        thr.join()


if __name__ == "__main__":
    features = get_audio_features("audio/alex_en.wav", extra_features=["delta", "delta2", "pitch"])
    print(features.shape)
