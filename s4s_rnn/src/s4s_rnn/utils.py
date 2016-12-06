import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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
        padding = np.repeat ([input_array[:1, :]], time_steps - i, axis=1)
        sample = np.append(padding, [input_array[:i, :]], axis=1)
        result = sample if result is None else np.append(result, sample, axis=0)
        pass
    for i in range(len(input_array) + 1 - time_steps):
        sample = input_array[i:i + time_steps, :]
        result = np.array([sample]) if result is None else np.append(result, [sample], axis=0)
        pass

    return result


def get_data_from_sessions(sessions, num_timesteps, output_dim=1, normalize=True, return_norm=False):
    """

    :param sessions: list of sweat4science.messages.Session object
    :param num_timesteps: number of lookback time steps, second dimension of Tensorflow shape
    :param output_dim: dimension of output data
    :param normalize: will normalize data if True
    :param return_norm: will return normalization result if True
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
        scaler = MinMaxScaler(feature_range=(0, 1))
        pass

    data_x = None
    data_y = None
    for data in data_multiple_arrays:
        if normalize:
            data = scaler.fit_transform(data)
            pass

        data_x_, data_y_ = data[:, :-output_dim], data[:, -output_dim:]
        data_x_ = reshape_array_by_time_steps(data_x_, time_steps=num_timesteps)

        data_x = data_x_ if data_x is None else np.append(data_x, data_x_, axis=0)
        data_y = data_y_ if data_y is None else np.append(data_y, data_y_, axis=0)
        pass
    if return_norm and normalize:
        return data_x, data_y, scaler
    else:
        return data_x, data_y


def evaluate_model(model, weights_file, data_x, data_y, scaler, horizon=None):
    """
    Predict output using given model and

    :param model: Keras model for prediction
    :param weights_file: H5 file containing weights. If None will skip loading weights and compiling
    :param data_x: input data
    :param data_y: actual output data
    :param scaler: MinMaxScaler object for unnormalizing
    :param horizon: time horizon for prediction, run full simulation if None
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
        data_y = data_y[-horizon:]
        pass

    # Run prediction
    prediction = model.predict(data_x)

    # Unnormalize and calculate error
    padding = np.zeros((len(data_y), data_x.shape[1] - 1))
    data_y_unnormed = scaler.inverse_transform(np.append(padding, data_y, axis=1))[:, -1]
    prediction_unnormed = scaler.inverse_transform(np.append(padding, prediction, axis=1))[:, -1]

    mse = np.mean((prediction_unnormed - data_y_unnormed)**2)

    return data_y_unnormed, prediction_unnormed, mse


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
        subplots[i].set_ylim([-1.1, 1.1])
        subplots[i].grid()
        pass

    figure.subplots_adjust(hspace=0.2)
    figure.set_size_inches((10, 10))
    plt.setp([a.get_xticklabels() for a in figure.axes[:-1]], visible=False)
    plt.show()
    return


def plot_predictions(predictions, targets, file_name,
                     title, y_label="Heart rate (hbm)", x_label="Time steps",
                     save_plot=False, show_plot=True):
    """
    Visualise comparison between prediction and actual data

    :param predictions: predicted outputs
    :param targets: actual outputs
    :param file_name: name of image file for saving plot
    :param title:
    :param y_label:
    :param x_label:
    :param save_plot: if True will write plot image to file_name
    :param show_plot: if True will show plot
    :return: None
    """
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    line1, = plt.plot(predictions, '-or', label='Predictions')
    line2, = plt.plot(targets, '-+g', label='Actual outputs')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(handles=[line1, line2], loc='upper center',
              bbox_to_anchor=(0.5, -0.08), fancybox=True,
              shadow=True, ncol=2)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()

    if save_plot and type(file_name).__name__ == 'str':
        plt.savefig(file_name)
        pass

    if show_plot:
        plt.show()
        pass
    pass

