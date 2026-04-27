import numpy as np


def grad(function, point, h = 1e-5, method = 'central'):
    if method not in ['central', 'forward']:
        raise ValueError("Invalid method. Supported methods are: 'central', 'forward'.")

    point = np.asarray(point, dtype = float)

    if point.ndim != 1:
        raise ValueError("grad expects a 1D array")

    base_value = function(point)

    if np.ndim(base_value) != 0:
        raise ValueError("grad expects function(point) to return a scalar")

    gradient = np.zeros_like(point, dtype = float)

    for index in range(len(point)):
        forward_point = point.copy()
        forward_point[index] += h

        if method == 'forward':
            gradient[index] = (function(forward_point) - base_value) / h

        elif method == 'central':
            backward_point = point.copy()
            backward_point[index] -= h
            gradient[index] = (function(forward_point) - function(backward_point)) / (2 * h)

    return gradient


def jacobian(function, point, h=1e-5, method='central'):
    if method not in ['central', 'forward']:
        raise ValueError("Invalid method. Supported methods are: 'central', 'forward'.")

    point = np.asarray(point, dtype=float)

    if point.ndim != 1:
        raise ValueError("jacobian expects a 1D array")

    base_output = np.asarray(function(point), dtype=float)

    if base_output.ndim != 1:
        raise ValueError("jacobian expects function(point) to return a 1D array")

    output_size = len(base_output)
    input_size = len(point)

    jacobian_matrix = np.zeros((output_size, input_size), dtype = float)

    for index in range(input_size):
        forward_point = point.copy()
        forward_point[index] += h

        if method == 'forward':
            forward_output = np.asarray(function(forward_point), dtype = float)
            jacobian_matrix[:, index] = (forward_output - base_output) / h

        elif method == 'central':
            backward_point = point.copy()
            backward_point[index] -= h

            forward_output = np.asarray(function(forward_point), dtype = float)
            backward_output = np.asarray(function(backward_point), dtype = float)

            jacobian_matrix[:, index] = (forward_output - backward_output) / (2 * h)

    return jacobian_matrix