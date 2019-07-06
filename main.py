import os
from pathlib import Path

import numpy as np

from extract_features import get_audio_features
from gender_model import lstm_gender_model
from age_model import lstm_age_model
from language_model import lstm_lang_model
from utils import norm_multiple
from file_io import get_data_files

gender_labels = {
    0: "female",
    1: "male"
}

age_labels = {
    0: "fifties",
    1: "fourties",
    2: "sixties",
    3: "teens",
    4: "thirties",
    5: "twenties"
}

lang_labels = {
    0: "english",
    1: "french",
    2: "german"
}

data_path = "audio/"
models_path = "model/"


def get_gender(out_data):
    out_data = out_data[0]
    return gender_labels[int(np.argmax(out_data))]


def get_age(out_data):
    out_data = out_data[0]
    return age_labels[int(np.argmax(out_data))]


def get_lang(out_data):
    out_data = out_data[0]
    return lang_labels[int(np.argmax(out_data))]


def main_program():

    gender_weights, gender_means, gender_stddev = get_data_files(models_path, "gender", 10)
    age_weights, age_means, age_stddev = get_data_files(models_path, "age", 30)
    lang_weights, lang_means, lang_stddev = get_data_files(models_path, "lang", 20)
    np.set_printoptions(precision=3)

    num_gender_labels = len(gender_labels)
    num_age_labels = len(age_labels)
    num_lang_labels = len(lang_labels)

    # declare the models
    gender_model = lstm_gender_model(num_gender_labels)
    age_model = lstm_age_model(num_age_labels)
    lang_model = lstm_lang_model(num_lang_labels)

    # load models
    gender_model.load_weights(gender_weights)
    age_model.load_weights(age_weights)
    lang_model.load_weights(lang_weights)

    mean_paths = [gender_means, age_means, lang_means]
    stddev_paths = [gender_stddev, age_stddev, lang_stddev]

    data_files = os.listdir(data_path)

    for data_file in data_files:
        data = get_audio_features(Path(data_path + data_file),
                                  extra_features=["delta", "delta2", "pitch"])
        data = np.array([data.T])

        data = norm_multiple(data, mean_paths, stddev_paths)

        gender_predict = gender_model.predict(data[0])
        age_predict = age_model.predict(data[1])
        lang_predict = lang_model.predict(data[2])
        
        gender_print = "{} ==> GENDER(lstm): {} gender_prob: {}".format(data_file,
                        get_gender(gender_predict).upper(), gender_predict)
        age_print = "{} ==> AGE(lstm): {} age_prob: {}".format(data_file,
                        get_age(age_predict).upper(), age_predict)
        lang_print = "{} ==> LANG(lstm): {} lang_prob: {}".format(data_file,
                        get_lang(lang_predict).upper(), lang_predict)
        
        print('=' * max(len(gender_print), len(age_print), len(lang_print)))
        print()
        print(gender_print)
        print(age_print)
        print(lang_print)
        print()
        print('=' * max(len(gender_print), len(age_print), len(lang_print)))


if __name__ == '__main__':
    main_program()
