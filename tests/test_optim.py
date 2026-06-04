import numpy as np
import pytest

from numcompute.optim import grad, jacobian


def test_grad_central_quadratic_function():
    def quadratic_function(point):
        return point[0] ** 2 + point[1] ** 2

    point = np.array([3.0, 4.0])
    result = grad(quadratic_function, point, method='central')
    expected = np.array([6.0, 8.0])

    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_grad_forward_quadratic_function():
    def quadratic_function(point):
        return point[0] ** 2 + point[1] ** 2

    point = np.array([3.0, 4.0])
    result = grad(quadratic_function, point, method='forward')
    expected = np.array([6.0, 8.0])

    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-4)


def test_grad_invalid_method():
    def sum_of_squares_function(point):
        return np.sum(point ** 2)

    with pytest.raises(ValueError):
        grad(sum_of_squares_function, np.array([1.0, 2.0]), method='wrong')


def test_grad_input_not_modified():
    def sum_of_squares_function(point):
        return np.sum(point ** 2)

    point = np.array([3.0, 5.0])
    original_point = point.copy()

    grad(sum_of_squares_function, point)

    assert np.array_equal(point, original_point)


def test_jacobian_central_vector_function():
    def vector_function(point):
        return np.array([
            point[0] + point[1],
            point[0] * point[1]
        ])

    point = np.array([3.0, 5.0])
    result = jacobian(vector_function, point, method='central')

    expected = np.array([
        [1.0, 1.0],
        [5.0, 3.0]
    ])

    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_forward_vector_function():
    def vector_function(point):
        return np.array([
            point[0] + point[1],
            point[0] * point[1]
        ])

    point = np.array([3.0, 5.0])
    result = jacobian(vector_function, point, method='forward')

    expected = np.array([
        [1.0, 1.0],
        [5.0, 3.0]
    ])

    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-4)


def test_jacobian_invalid_method():
    def identity_vector_function(point):
        return np.array([point[0], point[1]])

    with pytest.raises(ValueError):
        jacobian(identity_vector_function, np.array([1.0, 2.0]), method='wrong')


def test_jacobian_input_not_modified():
    def squared_vector_function(point):
        return np.array([point[0] ** 2, point[1] ** 2])

    point = np.array([3.0, 5.0])
    original_point = point.copy()

    jacobian(squared_vector_function, point)

    assert np.array_equal(point, original_point)