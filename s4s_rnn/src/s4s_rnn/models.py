from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


def create_model(model_name, hidden_neurons, input_dim, output_dim, input_shape=None):
    model = None
    if model_name == 'lstm':
        model = create_model_lstm(hidden_neurons, input_dim, output_dim, input_shape)
        pass
    elif model_name == 'gru':
        pass
    return model


def create_model_lstm(hidden_neurons, input_dim, output_dim, input_shape=None):
    model = Sequential()
    if input_shape is None:
        model.add(LSTM(hidden_neurons, input_dim=input_dim, return_sequences=False))
        pass
    else:
        model.add(LSTM(hidden_neurons, input_shape=input_shape, return_sequences=False))
        pass

    model.add(Dense(output_dim, input_dim=hidden_neurons))
    model.add(Activation('linear'))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model