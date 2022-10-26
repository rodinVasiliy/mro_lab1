import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show, imshow
from scipy.special import erf

O = [[0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0, 0]]

P = [[0, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0]]


def generate_binary_vector(pattern_vector, N, p):
    binary_vectors_array = []
    size_vector = np.shape(pattern_vector)
    for k in range(0, N):
        binary_vector = np.zeros(size_vector)
        for i in range(0, size_vector[0]):
            for j in range(0, size_vector[1]):
                u = random.random()
                if u < p:
                    binary_vector[i][j] = 1 - pattern_vector[i][j]
                else:
                    binary_vector[i][j] = pattern_vector[i][j]
        binary_vectors_array.append(binary_vector)
    return binary_vectors_array

#
# def print_vectors(vectors_array):
#     for vector in vectors_array:
#         print(vector)


def print_vector(vector):
    for row in vector:
        print(row)


def get_p_equal_1(vectors):
    size = np.shape(vectors)
    matrixP = np.sum(vectors, axis=0) / size[0]
    return matrixP


def bayesian_binary_classifier(vector, class_0, class_1, P_0, P_1, class_0_name, class_1_name):
    size = np.shape(vector)
    w01 = np.zeros(size)
    p0 = get_p_equal_1(class_0)
    p1 = get_p_equal_1(class_1)
    L = 0
    lymbda = 0
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            w01[i][j] = np.log(p0[i][j] / (1 - p0[i][j]) * (1 - p1[i][j]) / p1[i][j])
            L += vector[i][j] * w01[i][j]
            lymbda += np.log((1 - p1[i][j]) / (1 - p0[i][j]))
    lymbda += np.log(P_1 / P_0)
    if L > lymbda:
        return class_0_name
    return class_1_name


def get_binary_m(class_0, class_1):
    size = np.shape(class_1)
    arr_M = [0, 0]
    p0 = get_p_equal_1(class_0)
    p1 = get_p_equal_1(class_1)
    for i in range(0, size[1]):
        for j in range(0, size[2]):
            arr_M[0] += np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j]) * p0[i][j]
            arr_M[1] += np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j]) * p1[i][j]
    return arr_M


def get_binary_D(class_0, class_1):
    size = np.shape(class_1)
    arr_D = [0, 0]
    p0 = get_p_equal_1(class_0)
    p1 = get_p_equal_1(class_1)
    for i in range(0, size[1]):
        for j in range(0, size[2]):
            arr_D[0] += np.square(np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j])) * p0[i][j] * (
                    1 - p0[i][j])
            arr_D[1] += np.square(np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j])) * p1[i][j] * (
                    1 - p1[i][j])
    return arr_D


def get_laplas_function(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def calc_errors(class_0, class_1, P0, P1):
    p = [0, 0]
    lymbda = np.log(P0 / P1)
    M = get_binary_m(class_0, class_1)
    D = get_binary_D(class_0, class_1)
    p[0] = 1 - get_laplas_function((lymbda - M[0]) / np.sqrt(D[0]))
    p[1] = get_laplas_function((lymbda - M[1]) / np.sqrt(D[1]))
    return p


def get_experiment_errors(exp_classes, classes, P_0, P_1, class_name, names):
    exp_p = 0
    for vector in exp_classes:
        if bayesian_binary_classifier(vector, classes[0], classes[1], P_0, P_1, names[0], names[1]) != class_name:
            exp_p += 1
    print(f"amount of invalid vectors {class_name}: {exp_p}")
    exp_p /= len(exp_classes)
    # TODO подумать
    e = -1
    if exp_p != 0:
        e = np.sqrt((1 - exp_p) / (exp_p * len(exp_classes)))
    return exp_p, e


# def calc_w01(class0, class1):
#     size = np.shape(class1)
#     w01 = np.zeros((size[1], size[2]))
#     p0 = get_p_equal_1(class0)
#     p1 = get_p_equal_1(class1)
#     for i in range(0, size[1]):
#         for j in range(0, size[2]):
#             w01[i][j] = np.log(p0[i][j] / (1 - p0[i][j]) * (1 - p1[i][j]) / p1[i][j])
#     return w01


def find_invalid_vectors(exp_class, classes, P_0, P_1, className, names):
    invalid_vectors = []
    for vector in exp_class:
        if bayesian_binary_classifier(vector, classes[0], classes[1], P_0, P_1, names[0],
                                      names[1]) != className:
            invalid_vectors.append(vector)
    return invalid_vectors


if __name__ == '__main__':
    vectors_o = generate_binary_vector(O, 200, 0.3)
    vectors_p = generate_binary_vector(P, 200, 0.3)

    get_p_equal_1(vectors_o)
    class_of_vector_o = bayesian_binary_classifier(vectors_p[0], vectors_o, vectors_p, 0.5, 0.5, "O", "P")
    print(f"this vector from class {class_of_vector_o}")

    p = calc_errors(vectors_o, vectors_p, 0.5, 0.5)
    print(f"p_01 = {p[0]} p_10 = {p[1]}")

    exp_error, epsilon = get_experiment_errors(vectors_o, [vectors_p, vectors_o], 0.5, 0.5, "O", ["P", "O"])
    print(f"experimental error is {exp_error} epsilon is {epsilon}")

    imshow(np.array(O), cmap='gray')
    show()

    imshow(vectors_o[0])
    show()

    imshow(np.array(P), cmap='gray')
    show()

    imshow(vectors_p[0])
    show()

    invalid_vectors_from_o = find_invalid_vectors(vectors_o, [vectors_p, vectors_o], 0.5, 0.5, "O", ["P", "O"])
    if len(invalid_vectors_from_o) != 0:
        print('we find an error in vectors o')
        invalid_vectors_o = invalid_vectors_from_o[0]
        imshow(np.array(invalid_vectors_o))
        show()

    invalid_vectors_from_p = find_invalid_vectors(vectors_p, [vectors_p, vectors_o], 0.5, 0.5, "P", ["P", "O"])
    if len(invalid_vectors_from_p) != 0:
        print('we find an error in vectors p')
        invalid_vectors_p = invalid_vectors_from_p[0]
        imshow(np.array(invalid_vectors_p))
        show()