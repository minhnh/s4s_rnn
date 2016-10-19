#!/usr/bin/env python

import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import CSVLogger

import sweat4science as s4s
from sweat4science.workspace.Workspace import Workspace

workspace_folder = "/home/mnguy12s/rnd/session-data"
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
train_data_x = train_data_x.reshape((train_data_x.shape[0], 1, train_data_x.shape[1]))
test_data_x = test_data_x.reshape((test_data_x.shape[0], 1, test_data_x.shape[1]))

print("constructing LSTM model...")
input_dim = train_data_x.shape[2]
output_dim = train_data_y.shape[1]
hidden_neurons = 400
model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=input_dim, return_sequences=False))
model.add(Dense(output_dim, input_dim=hidden_neurons))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

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
