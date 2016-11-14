import numpy as np


def reshape_array_by_time_steps(input_array, time_steps=1):
    """
    Reshape training array into 3D tensor shape (nb_samples, timesteps, input_dim)
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
    for i in range(len(input_array) + 1 - time_steps):
        sample = input_array[i:i + time_steps, :]
        result = np.array([sample]) if result is None else np.append(result, [sample], axis=0)
        pass

    return result


def get_data_from_sessions(sessions, num_timesteps, output_dim=1, normalize=True, return_norm=False):
    """

    :param sessions:
    :param num_timesteps:
    :param output_dim:
    :param normalize:
    :param return_norm:
    :return:
    """
    train_data_x = None
    train_data_y = None
    for s in sessions:
        train_data_x_, train_data_y_ = get_data_from_session(s, output_dim=output_dim,
                                                             normalize=normalize,
                                                             return_norm=return_norm)
        train_data_x_ = reshape_array_by_time_steps(train_data_x_, time_steps=num_timesteps)
        train_data_y_ = train_data_y_[-len(train_data_x_):]
        train_data_x = train_data_x_ if train_data_x is None else \
            np.append(train_data_x, train_data_x_, axis=0)
        train_data_y = train_data_y_ if train_data_y is None else \
            np.append(train_data_y, train_data_y_, axis=0)
        pass
    return train_data_x, train_data_y


def get_data_from_session(session, output_dim=1, normalize=True, return_norm=False):
    """
    Create numpy data arrays from sweat4science.messages.Session objects
    :param session: sweat4science.messages.Session object
    :param output_dim: optional, number of output dimensions
    :return: arrays of input and output
    """
    data = np.array([session.velocity, session.time, session.distance,
                     session.acceleration, session.hbm], ndmin=2).T

    if output_dim > len(data[0]):
        return None

    normalization = None
    if normalize:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data - data_mean) / data_std
        normalization = (data_mean, data_std)
        pass

    if return_norm:
        return data[:, :-output_dim], data[:, -output_dim:], normalization
    else:
        return data[:, :-output_dim], data[:, -output_dim:]
