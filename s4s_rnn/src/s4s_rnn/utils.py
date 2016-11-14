import numpy as np


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

    data_mean = None
    data_std = None
    if normalize:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        pass

    data_x = None
    data_y = None
    for data in data_multiple_arrays:
        if normalize:
            data = (data - data_mean) / data_std
            pass

        data_x_, data_y_ = data[:, :-output_dim], data[:, -output_dim:]
        data_x_ = reshape_array_by_time_steps(data_x_, time_steps=num_timesteps)

        data_x = data_x_ if data_x is None else np.append(data_x, data_x_, axis=0)
        data_y = data_y_ if data_y is None else np.append(data_y, data_y_, axis=0)
        pass
    if return_norm and normalize:
        return data_x, data_y, data_mean, data_std
    else:
        return data_x, data_y
