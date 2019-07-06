
import keras_metrics as km

from keras import Sequential
from keras.layers import LSTM, Dense

from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 41  # 39


def lstm_age_model(num_labels):
    model = Sequential()
    model.add(LSTM(128 * 2, input_shape=(35, NUM_FEATURES), return_sequences=True, dropout=0.3))
    model.add(LSTM(128 * 2, dropout=0.3))
    model.add(Dense(128 * 2, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', km.categorical_precision()])
    return model


def main_class_age_train():
    dataset = "age_data_clean"  # good dataset #41(39 mfcc + pitch + magnitude)
    model = "model/lstm_age_"
    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_age_model, train_deepnn,
                      num_epoch_start=30,
                      num_features=NUM_FEATURES,
                      file_prefix="age")


if __name__ == '__main__':
    main_class_age_train()
