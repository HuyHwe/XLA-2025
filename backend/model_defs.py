"""Model layer definitions required for loading the pickled CNN model."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate2d


class Convolution:
    """Simple 2D convolution layer that mirrors the training-time implementation."""

    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape

        self.filter_shape = (num_filters, filter_size, filter_size)
        self.output_shape = (num_filters, input_height, input_width)

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_data):
        self.input_data = input_data
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            output[i] = correlate2d(self.input_data, self.filters[i], mode="same")

        output = np.maximum(output, 0)
        return output

    def backward(self, dL_out, lr):
        dL_input = np.zeros_like(self.input_data)
        dL_filters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            dL_filters[i] = correlate2d(self.input_data, dL_out[i], mode="valid")
            dL_input += correlate2d(dL_out[i], self.filters[i], mode="same")

        self.filters -= lr * dL_filters
        self.biases -= lr * dL_out

        return dL_input


class MaxPool:
    """Max pooling layer using the same logic as the notebook implementation."""

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = input_data[c, start_i:end_i, start_j:end_j]
                    self.output[c, i, j] = np.max(patch)

        return self.output

    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)

                    dL_dinput[c, start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        return dL_dinput


class Fully_Connected:
    """Fully connected layer paired with softmax output, as defined in training."""

    def __init__(self, input_size, output_size=10):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)

    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        _ = np.log(sum_exp_values)  # retained for parity with original implementation
        probabilities = exp_values / sum_exp_values

        return probabilities

    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)

    def forward(self, input_data):
        self.input_data = input_data
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases
        self.output = self.softmax(self.z)
        return self.output

    def backward(self, dL_out, lr):
        dL_y = np.dot(self.softmax_derivative(self.output), dL_out)
        dL_w = np.dot(dL_y, self.input_data.flatten().reshape(1, -1))
        dL_db = dL_y

        dL_input = np.dot(self.weights.T, dL_y)
        dL_input = dL_input.reshape(self.input_data.shape)

        self.weights -= lr * dL_w
        self.biases -= lr * dL_db

        return dL_input


def cross_entropy_loss(predictions, targets):
    num_samples = 10
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss


def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples
    return gradient


def predict(input_sample, conv, pool, full):
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    flattened_output = pool_out.flatten()
    predictions = full.forward(flattened_output)
    return predictions


