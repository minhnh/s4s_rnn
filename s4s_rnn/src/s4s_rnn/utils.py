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


def get_data_from_sessions(sessions, output_dim=1):
    """
    Create numpy data arrays from sweat4science.messages.Session objects
    :param sessions: list of sweat4science.messages.Session objects
    :param output_dim: optional, number of output dimensions
    :return: arrays of input and output
    """
    data = None
    for s in sessions:
        sample = np.array([s.velocity, s.slope, s.acceleration, s.hbm])
        data = sample if data is None else np.append(data, sample, axis=1)
        pass
    data = data.transpose()

    if output_dim > len(data[0]):
        return None

    return data[:, :-output_dim], data[:, -output_dim:]
