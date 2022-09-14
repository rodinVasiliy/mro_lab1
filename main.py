# This is a sample Python script.
import random

import numpy as np
import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def calculate_transformation_matrix(B):
    A = np.zeros((2, 2))
    A[0, 0] = np.sqrt(B[0, 0])
    A[0, 1] = 0
    A[1, 0] = B[0, 1] / np.sqrt(B[0, 0])
    A[1, 1] = np.sqrt(B[1, 1] - B[0, 1] * B[0, 1] / B[0, 0])
    return A


def get_normal_random_vector(n, mu=0, sigma=1):
    return np.matrix(np.random.normal(mu, sigma, n)).T


def get_training_sample(M, B):
    n, m = B.shape
    A = calculate_transformation_matrix(B)
    E = get_normal_random_vector(n)
    return A * E + M


def get_training_samples(M, B, N):
    samples = None
    for i in range(N):
        new_sample = get_training_sample(M, B)
        if samples is None:
            samples = new_sample
        else:
            samples = np.concatenate((samples, new_sample), axis=1)
    return samples.T


def show_vector_points(X, color='red'):
    for x in X:
        plt.scatter(x[0, 0], x[0, 1], color=color)


def get_estimate_expectation(samples):
    return (sum(samples) / len(samples)).T


def get_estimate_corr_matrix(samples):
    M_estimate = get_estimate_expectation(samples)
    return (sum([sample.T * sample for sample in samples]) / len(samples)) - M_estimate * M_estimate.T


def get_B_distance(M0, B0, M1, B1):
    B_half_sum = (B1 + B0) / 2
    return (
            0.25 * (M1 - M0).T * np.linalg.inv(B_half_sum) * (M1 - M0)
            + 0.5
            * np.log(
        np.linalg.det(B_half_sum)
        / np.sqrt(np.linalg.det(B1) * np.linalg.det(B0))
    )
    ).item()


def get_M_distance(M0, M1, B):
    M_diff = M1 - M0
    return (M_diff.T * np.linalg.inv(B) * M_diff).item()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    M1 = np.array([-1, 0]).reshape(2, 1)
    M2 = np.array([1, -1]).reshape(2, 1)
    M3 = np.array([-1, 2]).reshape(2, 1)
    R1 = np.array(([0.014, 0], [0, 0.020]))
    R2 = np.array(([0.031, 0], [0, 0.016]))
    R3 = np.array(([0.007, 0], [0, 0.020]))
    # first task
    samples1 = get_training_samples(M1, R1, 200)
    samples2 = get_training_samples(M2, R1, 200)
    show_vector_points(samples1)
    show_vector_points(samples2, color='blue')
    plt.show()

    # second task
    samples4 = get_training_samples(M2, R2, 200)
    samples5 = get_training_samples(M3, R3, 200)
    show_vector_points(samples1)
    show_vector_points(samples4, color='blue')
    show_vector_points(samples5, color='green')
    plt.show()
