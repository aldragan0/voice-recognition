import librosa
import numpy


def extract_max(pitches, magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(numpy.max(pitches[:, i]))
        new_magnitudes.append(numpy.max(magnitudes[:, i]))
    return numpy.asarray(new_pitches), numpy.asarray(new_magnitudes)


def smooth(x, window_len=11, window='hanning'):
    if window_len < 3:
        return x
    s = numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len:- window_len + 1]


def analyse(y, sr, fmin, fmax):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, S=None, fmin=fmin,
                                                fmax=fmax, threshold=0.75)
    shape = numpy.shape(pitches)
    pitches, magnitudes = extract_max(pitches, magnitudes, shape)
    return pitches, magnitudes


def get_pitch_magnitude(audio_data_path, sample_rate):

    duration = librosa.get_duration(filename=str(audio_data_path))
    y, sr = librosa.load(audio_data_path, sr=sample_rate, duration=duration, mono=True)
    pitches, magnitudes = analyse(y, sr, fmin=80, fmax=250)

    return y, pitches, magnitudes
