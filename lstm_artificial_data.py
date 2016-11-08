#!/usr/bin/env python3
import os
import time
import numpy as np
import random
from keras.callbacks import CSVLogger
from s4s_rnn import keras_lstm, utils

import matplotlib.pyplot as plt


def generate_sinx_plus_x(num_samples=1000, x1_max=200, x2_max=50):
    data_x = np.arange(0, x1_max, x1_max/num_samples)
    data_x = data_x.reshape((num_samples, 1))
    data_x = np.append(data_x,
                       np.arange(0, x2_max, x2_max/num_samples).reshape((num_samples, 1)),
                       axis=1)
    data_y = 5*np.sin(data_x[:, 0]) + data_x[:, 1]
    # add noise
    random.seed()
    for i in range(len(data_y)):
        data_y[i] += random.gauss(0, 1)
        pass
    return data_x, data_y.reshape((num_samples, 1))


def main():
    data = np.genfromtxt("artificial_data/sinx_plus_x.csv", delimiter=',')
    num_train = int(0.9*len(data))
    train_data_x_ = data[:num_train, :-1]
    train_data_y_ = data[:num_train, -1:]
    test_data_x_ = data[num_train:, :-1]
    test_data_y_ = data[num_train:, -1:]

    print("\nconstructing LSTM model...")
    input_dim = 2
    output_dim = 1
    hidden_neurons = 400
    num_epoch = 20
    for ntsteps in [5, 10, 15]:
        # create model
        model = keras_lstm.create_model(hidden_neurons, input_dim=None,
                                        input_shape=(ntsteps, input_dim),
                                        output_dim=output_dim)
        # Construct meaningful base name and write model to file
        base_name = "lstm_sinx_plus_x_" + str(ntsteps) + "step_" + str(input_dim) +\
                    "in_" + str(hidden_neurons) + "hidden_" + time.strftime("%Y%m%d") +\
                    "_" + str(num_epoch) + "epoch_"
        base_name = os.path.join("train_results", base_name)
        model_json = model.to_json()
        model_file_name = base_name  + "model.json"
        with open(model_file_name, "w") as json_file:
            json_file.write(model_json)
            pass

        train_data_x = utils.reshape_array_by_time_steps(train_data_x_, time_steps=ntsteps)
        train_data_y = train_data_y_[-len(train_data_x):]
        test_data_x = utils.reshape_array_by_time_steps(test_data_x_, time_steps=ntsteps)
        test_data_y = test_data_y_[-len(test_data_x):]

        model.compile(loss='mean_squared_error', optimizer='rmsprop')

        print("\nTraining...")
        csv_logger = CSVLogger(base_name + "training.log")
        model.fit(train_data_x, train_data_y, batch_size=(1),
                         nb_epoch=num_epoch, validation_data=(test_data_x, test_data_y),
                         callbacks=[csv_logger], verbose=2)

        # serialize weights to HDF5
        model.save_weights(base_name + "weights.h5")
        print("Saved model to disk")
        pass

    pass


if __name__ == "__main__":
    main()
    pass

