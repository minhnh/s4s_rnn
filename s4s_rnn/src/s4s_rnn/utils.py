import argparse
import os
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Standardization(object):
    def __init__(self):
        self.data_mean = None
        self.data_std = None
        return
    pass


class ReadableDir(argparse.Action):
    """
    Valid directory check from
    https://stackoverflow.com/questions/11415570/directory-path-types-with-argparse
    """
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("ReadableDir: {0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("ReadableDir: {0} is not a readable dir".format(prospective_dir))
        pass
    pass


def get_sessions():
    """  """
    from sweat4science.s4sconfig import workspace_dir
    from sweat4science.workspace.Workspace import Workspace
    from sweat4science.evaluation.sessionset import MF_sessionset as mfs

    workspace_folder = os.path.join(workspace_dir, "session-data")
    ws = Workspace(workspace_folder)
    sessions = mfs.ICT_indoor(ws)
    # Skip slope sessions
    for session in sessions:
        if len(re.findall("slope", str(session))) > 0:
            sessions.remove(session)
        pass
    return sessions


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
    num_column = normalized_data.shape[1]
    if scaler.__class__.__name__ == 'Standardization':
        return normalized_data * scaler.data_std[-num_column:] + scaler.data_mean[-num_column:]
    elif scaler.__class__.__name__ == 'MinMaxScaler':
        padding = np.zeros((len(normalized_data), len(scaler.data_range_) - num_column))
        return scaler.inverse_transform(np.append(padding, normalized_data, axis=1))[:, -num_column:]
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


def get_data_from_session(session):
    return np.array([session.distance, session.velocity, session.acceleration, session.time, session.hbm], ndmin=2).T


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
        data = get_data_from_session(s)
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
