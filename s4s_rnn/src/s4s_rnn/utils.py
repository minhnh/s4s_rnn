import numpy as np


def reshape_array_by_time_steps(input_array, time_steps=1):
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