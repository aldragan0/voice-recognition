
import keras_metrics as km

from keras import Sequential
from keras.layers import LSTM, Dense

from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 41  # 20, 41, 39


def lstm_gender_model(num_labels):
    model = Sequential()                # (num_features, timestamps)
    model.add(LSTM(100, input_shape=(35, NUM_FEATURES), dropout=0.3, return_sequences=True))
    model.add(LSTM(100, dropout=0.2))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', km.precision()])
    return model


def main_class_gender_train():
    dataset = "gender_data_clean"
    model = "model/lstm_gender_"
    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_gender_model, train_deepnn,
                      num_epoch_start=10,
                      num_features=NUM_FEATURES,
                      file_prefix="gender")


if __name__ == '__main__':
    main_class_gender_train()
