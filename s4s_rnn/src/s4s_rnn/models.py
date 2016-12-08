from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM, GRU


def create_model(model_name, hidden_neurons, input_dim, output_dim, input_shape=None):
    model = Sequential()
    if model_name == 'lstm':
        rnn_layer = LSTM
        pass
    elif model_name == 'gru':
        rnn_layer = GRU
        pass
    else:
        return None
    if input_shape is None:
        model.add(rnn_layer(hidden_neurons, input_dim=input_dim, return_sequences=False))
        pass
    else:
        model.add(rnn_layer(hidden_neurons, input_shape=input_shape, return_sequences=False))
        pass

    model.add(Dense(output_dim, input_dim=hidden_neurons))
    model.add(Activation('linear'))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model