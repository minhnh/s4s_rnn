#!/usr/bin/env python

import numpy as np

from keras.callbacks import CSVLogger

import sweat4science as s4s
from sweat4science.workspace.Workspace import Workspace

import keras_lstm

workspace_folder = "/home/minh/workspace/git/rnd/session-data"
ws = Workspace(workspace_folder)
user_name="MF83"
experiment_name = ["running_indoor_lactate_test", "running_indoor_session_01", "running_indoor_session_03$"]
session_number = None

sessions = ws.get(user_name, experiment_name, session_number)
sessions = sessions[0:3]

train_sessions = sessions[0:-1]
test_session = sessions[-1]

print("processing data...")
train_data = None

for ts in train_sessions:
    ts_train = np.array([ts.velocity, ts.slope, ts.acceleration, ts.hbm])
    train_data = ts_train if train_data is None else np.append(train_data, ts_train, axis=1)
    pass

train_data = train_data.transpose()
train_data_x = train_data[:,:-1]
train_data_y = train_data[:,-1:]

test_data = np.array([test_session.velocity, test_session.slope, test_session.acceleration, test_session.hbm]).transpose()
test_data_x = test_data[:,:-1]
test_data_y = test_data[:,-1:]

# reshape input to [samples, time steps, features], 1 timestep per sample
ntsteps = 5
train_cutoff = train_data.shape[0] - (train_data.shape[0] % ntsteps)
train_nsamples = int(train_cutoff/ntsteps)
test_cutoff = test_data.shape[0] - (test_data.shape[0] % ntsteps)
test_nsamples = int(test_cutoff/ntsteps)

train_data_x = train_data_x[:train_cutoff, :].reshape((train_nsamples, ntsteps, train_data_x.shape[1]))
train_data_y = train_data_y[:train_cutoff, :].reshape((train_nsamples, ntsteps * train_data_y.shape[1]))[:,-1:]
test_data_x = test_data_x[:test_cutoff, :].reshape((test_nsamples, ntsteps, test_data_x.shape[1]))
test_data_y = test_data_y[:test_cutoff, :].reshape((test_nsamples, ntsteps * test_data_y.shape[1]))[:,-1:]

print(train_data_x.shape)
print(train_data_y.shape)
print(test_data_x.shape)
print(test_data_y.shape)

print("constructing LSTM model...")
input_dim = train_data_x.shape[2]
output_dim = train_data_y.shape[1]
hidden_neurons = 400
model = keras_lstm.create_model(hidden_neurons, input_dim=None,
                                input_shape=(ntsteps, train_data_x.shape[2]), output_dim=output_dim)

print("training...")
csv_logger = CSVLogger('training.log')
model.fit(train_data_x, train_data_y, batch_size=(1),
          nb_epoch=200, validation_data=(test_data_x, test_data_y),
          callbacks=[csv_logger], verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights.h5")
print("Saved model to disk")