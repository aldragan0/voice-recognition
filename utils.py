
import numpy as np


def min_label_count(labels: dict):
    min_label, min_count = "", -1
    for key, value in labels.items():
        if min_count == -1 or min_count > value:
            min_count = value
            min_label = key
    return min_label, min_count


def transpose_vector(inputs):
    new_inputs = []
    for i in range(inputs.shape[0]):
        new_inputs.append(inputs[i].T)
    return np.array(new_inputs)


def get_count(labels):
    """
    count each label type
    :param labels: items to count
    :return: {label_type: freq}
    """
    ages = dict()
    for label in labels:
        if label in ages.keys():
            ages[label] += 1
        else:
            ages[label] = 1
    return ages


def get_mean_stddev(input_data: np.array):
    """
    get means and std_dev for each column in the input vector
    :param input_data:
    :return: vector containing means for each column
    """
    means = np.mean(input_data, axis=0)
    std_dev = np.std(input_data, axis=0)
    return means, std_dev


def normalize_data(input_data: np.array, means, std_dev):
    """
    normalize data by simple statistic data normalisation
    :param input_data: data to be normalized
    :param means:
    :param std_dev:
    :return: normalized data
    """
    norm_input_data = (input_data - means) / std_dev
    return norm_input_data


def labels_to_categorical(labels):
    """
    one-hot encodes the data in a given sequence
    :param labels: data to encode
    :return: one-hot representation of the data
    """
    norm_labels = []
    label_set = list(set(labels))
    label_set.sort()
    num_labels = len(label_set)

    for label in labels:
        for i in range(len(label_set)):
            if label == label_set[i]:
                label_arr = np.zeros(num_labels)
                label_arr[i] = 1
                norm_labels.append(label_arr)
    return np.array(norm_labels)


def norm_multiple(input_data: np.array, mean_paths, stddev_paths):
    """
    normalize data after multiple means, stddevs files

    :param input_data: data to be normalized
    :param mean_paths: list containing paths to the means
    :param stddev_paths: list containing paths to the stddevs
    :return: list of normalized data, where each is normalized 
             after a pair of (mean, stddev)
    """

    norm_input_data = []

    for mean_path, stddev_path in zip(mean_paths, stddev_paths):
        means = np.load(mean_path)
        stddevs = np.load(stddev_path)
        
        input_copy = np.copy(input_data)
        norm_input = normalize_data(input_copy, means, stddevs)
        norm_input_data.append(norm_input)

    return norm_input_data