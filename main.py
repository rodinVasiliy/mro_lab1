# This is a sample Python script.
import random

import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def calculate_transformation_matrix(correlation_matrix):
    result = np.zeros((2, 2))
    result[0, 0] = np.sqrt(correlation_matrix[0][0])
    result[0, 1] = 0
    result[1, 0] = correlation_matrix[0][1] / np.sqrt(correlation_matrix[0][0])
    result[1, 1] = np.sqrt(
        correlation_matrix[1][1] - correlation_matrix[0][1] * correlation_matrix[0][1] / correlation_matrix[0][0])
    return result


def get_training_samples(M, correlation_matrix, N):
    n = 2
    A = calculate_transformation_matrix(correlation_matrix)
    y1 = np.zeros((n, N))
    x1 = np.zeros((n, N))
    for i in range(0, n):
        for j in range(0, N):
            y1[i, j] = random.normalvariate(0, 1)
    print(y1)
    for k in range(0, n):
        for i in range(0, N):
            sum1 = 0
            for j in range(0, 2):
                sum1 = sum1 + A[k, j] * y1[j, i]
            x1[k, i] = sum1 + M[k]
    return x1


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    M1 = (-1, 0)
    M2 = (1, -1)
    M3 = (-1, 2)
    R1 = [[0.1, 0], [0, 0.1]]
    # A = np.zeros((2, 2))
    # A[0, 0] = np.sqrt(R1[0][0])
    # A[0, 1] = 0
    # A[1, 0] = R1[0][1] / np.sqrt(R1[0][0])
    # A[1, 1] = np.sqrt(R1[1][1] - R1[0][1] * R1[0][1] / R1[0][0])
    # print(A)
    # n = 2
    # N = 200
    # y1 = np.zeros((n, N))
    # x1 = np.zeros((n, N))
    # for i in range(0, n):
    #     for j in range(0, N):
    #         y1[i, j] = random.normalvariate(0, 1)
    # print(y1)
    # for k in range(0, n):
    #     for i in range(0, N):
    #         sum1 = 0
    #         for j in range(0, 2):
    #             sum1 = sum1 + A[k, j] * y1[j, i]
    #         x1[k, i] = sum1 + M1[k]
    x1 = get_training_samples(M1, R1, N=200)
    print("=========================================================================================")
    print("X1")
    print(x1)
