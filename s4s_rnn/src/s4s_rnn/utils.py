import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import MinMaxScaler


class Standardization(object):
    def __init__(self):
        self.data_mean = None
        self.data_std = None
        return
    pass


def reshape_array_by_time_steps(input_array, time_steps=1):
    """
    Reshape training array into 3D tensor shape (nb_samples, timesteps, input_dim)
    Beginning of input array is padded by repeating the first row
    :param input_array: 2D array of shape (nb_samples, input_dim)
    :param time_steps: number of lookback time steps given to train the recurrent model
    :return: reshaped array
    """
    if type(input_array).__name__ == 'ndarray':
        input_array = np.array(input_array)
        pass

    if len(input_array.shape) != 2:
        raise ValueError("Only accept 2D arrays as input")

    if type(time_steps).__name__ != 'int' or time_steps <= 0:
        raise ValueError('Invalid number of time steps')

    if (len(input_array) < time_steps):
        time_steps = len(input_array)
        pass

    result = None
    for i in range(1, time_steps):
        padding = np.repeat([input_array[:1, :]], time_steps - i, axis=1)
        sample = np.append(padding, [input_array[:i, :]], axis=1)
        result = sample if result is None else np.append(result, sample, axis=0)
        pass
    for i in range(len(input_array) + 1 - time_steps):
        sample = input_array[i:i + time_steps, :]
        result = np.array([sample]) if result is None else np.append(result, [sample], axis=0)
        pass

    return result


def unnormalize(normalized_data, scaler):
    """
    Unnormalize data using a MinMaxScaler object

    :param normalized_data:
    :param scaler: MinMaxScaler object, should contain additional var if
                   old_norm is true
    :param old_norm: whether to use old normalization strategy
    :return: None
    """
    if scaler.__class__.__name__ == 'Standardization':
        return (normalized_data * scaler.data_std[-1] + scaler.data_mean[-1]).flatten()
    elif scaler.__class__.__name__ == 'MinMaxScaler':
        padding = np.zeros((len(normalized_data), len(scaler.data_range_) - 1))
        return scaler.inverse_transform(np.append(padding, normalized_data, axis=1))[:, -1]
    else:
        raise ValueError("Unrecognized scaler type: %s" % scaler.__class__.__name__)


def normalize_with_scaler(data, scaler):
    """

    :param data:
    :param scaler:
    :return:
    """
    num_column = data.shape[1]
    if scaler.__class__.__name__ == 'Standardization':
        return (data - scaler.data_mean[-num_column:]) / scaler.data_std[-num_column:]
    elif scaler.__class__.__name__ == 'MinMaxScaler':
        padding = np.zeros((len(data), len(scaler.data_range_) - num_column))
        return scaler.transform(np.append(padding, data, axis=1))[:, -num_column:]
    else:
        raise ValueError("Unrecognized scaler type: %s" % scaler.__class__.__name__)


def get_scaler(data, old_norm, return_data=False):
    """
    :param data:
    :param old_norm: if true use standardization
    :return: normalization scaler
    """
    data_normed = None
    if old_norm:
        scaler = Standardization()
        scaler.data_mean = np.mean(data, axis=0)
        scaler.data_std = np.std(data, axis=0)
        if return_data:
            data_normed = normalize_with_scaler(data, scaler)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normed = scaler.fit_transform(data)
        pass

    if return_data:
        return scaler, data_normed
    else:
        return scaler


def get_data_from_sessions(sessions, num_timesteps=None, output_dim=1, normalize=True,
                           return_norm=False, old_norm=False):
    """
    Get data array from lists of sweat4science.messages.Session objects

    :param sessions: list of sweat4science.messages.Session object
    :param num_timesteps: number of lookback time steps, second dimension of Tensorflow shape,
                          if None will not call reshape_array_by_time_steps
    :param output_dim: dimension of output data
    :param normalize: will normalize data if True
    :param return_norm: will return normalization result if True
    :param old_norm: normalize using old technique if True
    :return: input data, output data and normalization results if return_norm is True
    """
    data_multiple_arrays = []
    data_single_array = None
    for s in sessions:
        data = np.array([s.distance, s.velocity, s.acceleration, s.time, s.hbm], ndmin=2).T
        data_multiple_arrays.append(data)
        data_single_array = data if data_single_array is None else \
            np.append(data_single_array, data, axis=0)
        pass

    scaler = None
    if normalize:
        scaler = get_scaler(data_single_array, old_norm=old_norm)
        pass

    data_x = None
    data_y = None
    for data in data_multiple_arrays:
        if normalize:
            data = normalize_with_scaler(data, scaler)
            pass

        data_x_, data_y_ = data[:, :-output_dim], data[:, -output_dim:]
        if num_timesteps is not None:
            data_x_ = reshape_array_by_time_steps(data_x_, time_steps=num_timesteps)
            pass

        data_x = data_x_ if data_x is None else np.append(data_x, data_x_, axis=0)
        data_y = data_y_ if data_y is None else np.append(data_y, data_y_, axis=0)
        pass

    if return_norm and normalize:
        return data_x, data_y, scaler
    else:
        return data_x, data_y


def evaluate_model(model, weights_file, data_x, data_y, horizon=None):
    """
    Predict output using given model and

    :param model: Keras model for prediction
    :param weights_file: H5 file containing weights. If None will skip loading weights and compiling
    :param data_x: input data
    :param data_y: actual output data
    :param scaler: MinMaxScaler or Standardization object for unnormalizing
    :param horizon: time horizon for prediction, run full simulation if None
    :param old_norm: normalize using old technique if True
    :return:
    """
    # Prepare model
    if weights_file is not None:
        model.load_weights(weights_file)
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
        pass

    # if horizon is not None train on [:-horizon] before predicting
    if horizon is not None:
        model.fit(data_x[:-horizon], data_y[:-horizon], batch_size=(1),
                  nb_epoch=1, validation_split=0.0, verbose=0)
        data_x = data_x[-horizon:]
        pass

    # Run prediction
    prediction = model.predict(data_x)

    return prediction


def plot_inputs(inputs):
    """
    Function to plot input features, each feature in a separated subplot
    :param inputs: input features as (num_samples, num_features) array
    :return: None
    """
    num_features = inputs.shape[1]
    feature_names = ['distance', 'velocity', 'acceleration', 'time']
    figure, subplots = plt.subplots(num_features, sharex=True)
    for i in range(num_features):
        subplots[i].plot(inputs[:, i], '-or', label=feature_names[i])
        subplots[i].set_title('Plot of normalized %s' % feature_names[i])
        max_input = np.max(inputs[:, i])
        min_input = np.min(inputs[:, i])
        difference = max_input - min_input
        subplots[i].set_ylim([min_input - 0.1*difference,
                              max_input + 0.1*difference])
        subplots[i].grid()
        pass

    figure.subplots_adjust(hspace=0.2)
    figure.set_size_inches((10, 10))
    plt.setp([a.get_xticklabels() for a in figure.axes[:-1]], visible=False)
    plt.show()
    return


def plot_predictions(predictions, prediction_names, true_output, title, file_name=None,
                     y_label="Heart rate (hbm)", x_label="Time steps", show_plot=True):
    """
    Visualise comparison between prediction and actual data

    :param predictions: list of predicted outputs
    :param prediction_names: names of predictions for plot labels
    :param true_output: actual outputs
    :param file_name: name of image file for saving plot
    :param title:
    :param y_label:
    :param x_label:
    :param save_plot: if True will write plot image to file_name
    :param show_plot: if True will show plot
    :return: None
    """
    if len(predictions) != len(prediction_names):
        print("Lengths of prediction list and prediction names must equal")
        return

    max_len = max(map(lambda pred : len(pred), predictions))
    x = list(range(max_len))

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    colors = cycle('rbgcmykw')
    lines = []
    line_actual, = plt.plot(x, true_output[-max_len:], '-o', c=next(colors), markersize=4,
                            label='True output')
    lines.append(line_actual)
    for index, prediction in enumerate(predictions):
        line_predict, = plt.plot(x[-len(prediction):], prediction, '-+',
                                 c=next(colors), markersize=4,
                                 label=prediction_names[index])
        lines.append(line_predict)
        pass

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(handles=lines, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), fancybox=True,
              shadow=True, ncol=2)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()

    if type(file_name).__name__ == 'str':
        plt.savefig(file_name)
        pass

    if show_plot:
        plt.show()
        pass
    pass

