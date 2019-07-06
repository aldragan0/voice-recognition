
import numpy as np


def write_to_file_features(output_file, features):
    """
    writes a data sample(matrix) to file
    write whole dataset to file:
        for i in range(dataset.shape[0]):
            write_to_file_features("example.txt", dataset[i])
    :param output_file: file to write to
    :param features: data sample
    """
    with open(output_file, 'a+') as file_stream:
        for f in features:
            for el in f:
                file_stream.write(str(el))
                file_stream.write(",")
        file_stream.write('\n')


def write_to_file_labels(output_file, vector):
    """
    write elements of a 1d vector to file
    :param output_file: output file
    :param vector: data to be written
    """
    with open(output_file, 'w+') as file:
        for item in vector:
            file.write(str(item))
            file.write('\n')


def features_from_file(input_file, num_features=20):
    """
    extract mfcc features from file
    :param input_file: feature file
    :param num_features: feature count
    :return: extracted features
    """
    features_matrix = []
    with open(input_file, 'r') as file_stream:
        for matrix in file_stream:
            matrix_str = matrix.strip("\n").split(",")
            matrix_float = [float(matrix_str[i]) for i in range(len(matrix_str) - 1)]
            matrix_float = np.array(matrix_float)
            matrix_float = matrix_float.reshape(num_features, 35)
            features_matrix.append(matrix_float)
    return np.array(features_matrix)


def labels_from_file(input_file):
    labels = []
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip('\n')
            labels.append(line)
    return labels


def get_data_files(filepath, prefix, num_epochs, num_features=41,
                   model_type='lstm'):
    """
    model folder of type: type_prefix_features
    model file of type: prefix_features_epochs.model
    means and stddev file of type: means/stddev_prefix_numfeatures.npy
    """
    num_epochs = str(num_epochs)
    num_features = str(num_features)

    model_name = '_'.join([model_type, prefix, num_features])
    model_file = model_name + '_' + num_epochs + ".model"
    model_path = filepath + model_name + "/"
    means_file = '_'.join(["means", prefix, num_features]) + ".npy"
    stddevs_file = '_'.join(["stddev", prefix, num_features]) + ".npy"

    means_file = model_path + means_file
    stddevs_file = model_path + stddevs_file
    model_file = model_path + model_file

    return model_file, means_file, stddevs_file


def add_history(filepath, history_train, history_valid, metrics):
    """
    add to history from metrics collected on train, test data
    :param filepath:
    :param metrics: metrics to save to the file
    :param history_train: dict containing training metrics per epoch
    :param history_valid: tuple containig validation metrics per epoch
    """
    for i in range(len(metrics)):
        with open(filepath + "_" + metrics[i], "a+") as file:
            file.write(str(history_train[metrics[i]][0]))
            file.write(" ")
            file.write(str(history_valid[i]))
            file.write('\n')


def load_metric(filepath):
    """
    load the metric data from a file
    :param filepath: file to store metric data
    :return: np array containing metric data of type (train, validation)
    """
    history = list()
    with open(filepath, 'r') as file:
        for line in file:
            values = [np.float(i) for i in line.strip(" \n").split(" ")]
            values = np.asarray(values)
            history.append(values)
    return np.asarray(history)


def concat_files(dirpath, filenames, out_file, lines_per_file=-1):
    """
    concatenate multiple files into a single one
    :param dirpath: path to the files
    :param filenames: the list of filenames to concatenate
    :param out_file: where to store the concatenated data
    :param lines_per_file: how many lines to take from each file
    """
    if dirpath[-1] != '/':
        dirpath = dirpath + '/'
    out_path = dirpath + out_file
    if lines_per_file == -1:
        lines_per_file = 2 ** 20
    with open(out_path, 'w') as outfile:
        for filename in filenames:
            count = 0
            file_path = dirpath + filename
            with open(file_path) as infile:
                for line in infile:
                    if line != "" and count < lines_per_file:
                        outfile.write(line)
                        count += 1
                    elif count >= lines_per_file:
                        break
