import numpy as np
from sklearn.model_selection import train_test_split

from file_io import features_from_file, labels_from_file, add_history
from utils import labels_to_categorical, transpose_vector, get_count, get_mean_stddev, normalize_data


BATCH_SIZE = 128


def train_deepnn(model_file, inputs, outputs, model, num_epochs):
    x_train, x_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, random_state=36)

    means, std_dev = get_mean_stddev(x_train)

    filepath = '/'.join(model_file.split("/")[:-1])
    filename = model_file.split("_")[2] + "_" + str(x_train.shape[2])

    np.save(filepath + "/means_" + filename + ".npy", means)
    np.save(filepath + "/stddev_" + filename + ".npy", std_dev)

    x_train = normalize_data(x_train, means, std_dev)
    x_valid = normalize_data(x_valid, means, std_dev)

    y_train = labels_to_categorical(y_train)
    y_valid = labels_to_categorical(y_valid)

    for epoch in range(num_epochs):
        history_train = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        history_valid = model.evaluate(x_valid, y_valid, verbose=0, batch_size=BATCH_SIZE)

        key_list = list(history_train.history.keys())
        score_train = history_train.history["loss"][0]
        acc_train = history_train.history["acc"][0]

        print()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print(" - loss: {:.4f} - acc: {:.4f}".format(score_train, acc_train))
        print()
        print("logloss score: %.4f" % history_valid[0])
        print("Validation set Accuracy: %.4f" % history_valid[1])
        add_history(model_file, history_train.history, history_valid, key_list)

    return model


def train_model(dataset, model_type, train_fn, model_file, **kwargs):

    num_epochs = kwargs['num_epochs']
    num_features = kwargs['num_features']
    file_prefix = kwargs['file_prefix']

    print("Loading dataset...")
    input_train_file = dataset + "/" + file_prefix + "_in"
    output_train_file = dataset + "/" + file_prefix + "_out"
    inputs = features_from_file(input_train_file, num_features)
    inputs = transpose_vector(inputs)
    outputs = labels_from_file(output_train_file)

    label_count = get_count(outputs)
    print(label_count)

    print("Finished loading dataset")

    print(inputs.shape)

    model = model_type(len(label_count))
    print("Training model..")
    model = train_fn(model_file, inputs, outputs, model, num_epochs=num_epochs)
    print("Done training model...")

    model.save(model_file + ".model")


def train_multi_epoch(dataset, filepath, model, train_fn, num_epoch_start, num_epoch_end=0,
                      delta_epochs=10, **kwargs):

    num_epoch_end = max(num_epoch_start, num_epoch_end) + delta_epochs
    print("Training on: " + dataset)
    print("Output file: " + filepath)

    model_name = filepath.split('/')[1]
    if model_name[-1] != '_':
        model_name += '_'
    if filepath[-1] != '/':
        filepath += '/'

    for epochs in range(num_epoch_start, num_epoch_end, delta_epochs):
        model_file = filepath + model_name + str(epochs)
        print("Model file: ", model_file)
        train_model(dataset, model, train_fn, model_file, **kwargs, num_epochs=epochs)
