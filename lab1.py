# This is a sample Python script.

import matplotlib.pyplot as plt
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def calculate_transformation_matrix(correlation_matrix):
    A = np.zeros((2, 2))
    A[0, 0] = np.sqrt(correlation_matrix[0, 0])
    A[0, 1] = 0
    A[1, 0] = correlation_matrix[0, 1] / np.sqrt(correlation_matrix[0, 0])
    A[1, 1] = np.sqrt(correlation_matrix[1, 1] - correlation_matrix[0, 1] ** 2 / correlation_matrix[0, 0])
    return A


def get_normal_random_vector(n):
    tmp_array = np.zeros((2, n))
    for i in range(0, n):
        tmp_array += np.random.uniform(0.5, -0.5, (2, n))
    tmp_array /= np.sqrt(n)
    tmp_array /= np.sqrt(1 / 12)
    return tmp_array


def show_vector_points(X, color='red'):
    for x in X:
        plt.scatter(x[0], x[1], color=color)


def show_vector_points1(X, color='red'):
    plt.scatter(X[0, :], X[1, :], color=color)


def get_training_sample(M, R, E):
    A = np.array(calculate_transformation_matrix(R))
    tmp = (np.dot(A, E)).reshape(2, 1)
    return tmp + M


def get_training_samples(M, R, N):
    samples = None
    E = get_normal_random_vector(N)
    for i in range(N):
        random_vector = E[:, i]
        new_sample = get_training_sample(M, R, random_vector)
        if samples is None:
            samples = new_sample
        else:
            samples = np.concatenate((samples, new_sample), axis=1)
    return samples


def get_estimate_expectation(samples):
    tmp = (np.sum(samples, axis=1)).reshape(2, 1)
    tmp = tmp / len(samples[0])
    return tmp


def get_estimate_corr_matrix(samples):
    M_estimate = get_estimate_expectation(samples)
    tmp = 0
    for i in range(0, len(samples[0]) - 1):
        A = samples[:, i].reshape(2, 1)
        R = samples[:, i].reshape(1, 2)
        tmp += np.dot(A, R)
    tmp /= len(samples[0])
    M_T = M_estimate.reshape(1, 2)
    tmp -= np.dot(M_estimate, M_T)
    return tmp


def get_B_distance(M0, R0, M1, R1):
    B_half_sum = (R1 + R0) / 2
    M1_minus_M0 = M1 - M0
    M1_minus_M0_T = M1_minus_M0.reshape(1, 2)
    det_B_half = np.linalg.det(B_half_sum)
    det_B0 = np.linalg.det(R0)
    det_B1 = np.linalg.det(R1)
    inverse_B_half = np.linalg.inv(B_half_sum)
    f = 0.25 * M1_minus_M0_T
    s = np.dot(f, inverse_B_half)
    t = np.dot(s, M1_minus_M0)
    result = t + 0.5 + np.log(det_B_half / (det_B1 * det_B0))
    return result.item()


def get_M_distance(M0, M1, R):
    M_diff = M1 - M0
    M_diff_T = M_diff.reshape(1, 2)
    f = np.dot(M_diff_T, np.linalg.inv(R))
    result = np.dot(f, M_diff)
    return result.item()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    M1 = np.array([-1, 0]).reshape(2, 1)
    M2 = np.array([1, -1]).reshape(2, 1)
    M3 = np.array([-1, 2]).reshape(2, 1)
    R1 = np.array(([0.4, 0.3], [0.3, 0.5]))
    R2 = np.array(([0.3, 0], [0, 0.3]))
    R3 = np.array(([0.87, -0.87], [-0.8, 0.95]))
    # first task
    samples1 = get_training_samples(M1, R1, 200)
    samples2 = get_training_samples(M2, R1, 200)
    show_vector_points1(samples1)
    show_vector_points1(samples2, color='blue')
    plt.show()
    B_distance_with_same_correlation_matrix = get_B_distance(M1, R1, M2, R1)
    M_distance_with_same_correlation_matrix = get_M_distance(M1, M2, R1)
    # second task
    samples3 = get_training_samples(M2, R2, 200)
    samples4 = get_training_samples(M3, R3, 200)
    show_vector_points1(samples1)
    show_vector_points1(samples3, color='blue')
    show_vector_points1(samples4, color='green')
    plt.show()

    math_expectation_estimate_s1 = get_estimate_expectation(samples1)
    math_expectation_estimate_s3 = get_estimate_expectation(samples3)
    math_expectation_estimate_s4 = get_estimate_expectation(samples4)
    corr_matrix_estimate_s1 = get_estimate_corr_matrix(samples1)
    corr_matrix_estimate_s3 = get_estimate_corr_matrix(samples3)
    corr_matrix_estimate_s4 = get_estimate_corr_matrix(samples4)
    B_distance_S1_S3 = get_B_distance(M1, R1, M2, R2)
    B_distance_S1_S4 = get_B_distance(M1, R1, M3, R3)
    B_distance_S3_S4 = get_B_distance(M2, R2, M3, R3)

    print("Math expectation M1: \n", M1, "\n")
    print("Estimate math expectation M1: \n", math_expectation_estimate_s1, "\n")
    print("Сorrelation matrix R1: \n", R1, "\n")
    print("Estimate correlation matrix R1: \n", corr_matrix_estimate_s1, "\n")

    print("Math expectation M2: \n", M2, "\n")
    print("Estimate math expectation M2: \n", math_expectation_estimate_s3, "\n")
    print("Сorrelation matrix R2: \n", R2, "\n")
    print("Estimate correlation matrix R2: \n", corr_matrix_estimate_s3, "\n")

    print("Math expectation M3: \n", M3, "\n")
    print("Estimate math expectation M3: \n", math_expectation_estimate_s4, "\n")
    print("Сorrelation matrix R3: \n", R3, "\n")
    print("Estimate correlation matrix R3: \n", corr_matrix_estimate_s4, "\n")

    print(
        "Bhatacharyas distance for vectors with same correlation matrix R1 and math expectations M1 и M2: \n",
        B_distance_with_same_correlation_matrix, "\n")
    print(
        "Mahalanobis distance for vectors with same correlation matrix R1 and math expectations M1 и M2: \n",
        M_distance_with_same_correlation_matrix, "\n")
    print(
        "Bhatacharyas distance for S1(M1, R1) and S3(M2, R2): \n",
        B_distance_S1_S3, "\n")
    print(
        "Bhatacharyas distance for S1(M1, R1) and S4(M3, R3): \n",
        B_distance_S1_S4, "\n")
    print(
        "Bhatacharyas distance for S3(M2, R2) and S4(M3, R3): \n",
        B_distance_S3_S4, "\n")
