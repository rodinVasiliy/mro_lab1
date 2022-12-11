import math

import numpy as np


def get_bayos_lines(Bj, Bl, Mj, Ml, Pl, Pj, k):
    d, e = __get_de(Mj, Ml, Bj, Bl)
    f = __get_f(Mj, Ml, Bj, Bl, Pl, Pj)
    a = __get_a(Bj, Bl)
    b = __get_b(Bj, Bl) * k + d
    c = __get_c(Bj, Bl) * (k ** 2) + e * k + f

    if not np.array_equal(Bj, Bl):
        D = (b ** 2) - 4 * a * c
        sqrt_D = np.sqrt(D)

        return (-b - sqrt_D) / (2 * a), (-b + sqrt_D) / (2 * a)
    return -1 / d * (e * k + f)


def __get_a(Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    return Bj_inv[0][0] - Bl_inv[0][0]


def __get_b(Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    return (Bj_inv[1][0] - Bl_inv[1][0]) + (Bj_inv[0][1] - Bl_inv[0][1])


def __get_c(Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    return Bj_inv[1][1] - Bl_inv[1][1]


def __get_de(Mj, Ml, Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    Mj_transpos = np.transpose(Mj)
    Ml_transpos = np.transpose(Ml)

    Ml_mul_Bl = np.matmul(Ml_transpos, Bl_inv)
    Mj_mul_Bj = np.matmul(Mj_transpos, Bj_inv)

    res = 2 * (Ml_mul_Bl - Mj_mul_Bj)
    return res[0], res[1]


def __get_f(Mj, Ml, Bj, Bl, pl, pj):
    # Слагаемое 1
    Bl_det = np.linalg.det(Bl)
    Bj_det = np.linalg.det(Bj)
    sum_part1 = np.log(Bl_det / Bj_det)

    # Слагаемое 2
    sum_part2 = 2 * np.log(pl / pj)

    # Слагаемое 3
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    Mj_transpos = np.transpose(Mj)
    Ml_transpos = np.transpose(Ml)

    MlBlMl = __mul_3_matrices(Ml_transpos, Bl_inv, Ml)
    MjBjMj = __mul_3_matrices(Mj_transpos, Bj_inv, Mj)

    return sum_part1 + sum_part2 - MlBlMl + MjBjMj


def get_baios_line(shape, b1, b2, m1, m2, p):
    step = 0.005
    points_count = shape[0] / step

    x0_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))
    x1_a_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))
    x1_b_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))

    counter = 0
    i = -(shape[0] / 2.0)
    if not np.array_equal(b1, b2):
        while i < shape[1] / 2.0 - step:
            x0_vector[counter] = i
            x1_vector = get_bayos_lines(b1, b2, m1, m2, p, p, i)
            x1_a_vector[counter] = x1_vector[0]
            x1_b_vector[counter] = x1_vector[1]
            counter += 1
            i += step
        return x0_vector, x1_a_vector, x1_b_vector
    while i < shape[1] / 2.0 - step:
        x0_vector[counter] = i
        x1_vector = get_bayos_lines(b1, b2, m1, m2, p, p, i)
        x1_a_vector[counter] = x1_vector
        counter += 1
        i += step
    return x0_vector, x1_a_vector


def __mul_3_matrices(mat_1, mat_2, mat_3):
    tmp_result = np.matmul(mat_1, mat_2)
    return np.matmul(tmp_result, mat_3)


# X классифицируется в класс с параметрами индекса 1 -> ошибка 0. Иначе 1
def baios_classifier(P1, P2, B1, B2, M1, M2, X):
    if __discriminant_function(P1, B1, M1, X) >= __discriminant_function(P2, B2, M2, X):
        return 0
    return 1


def __discriminant_function(P, B, M, X):
    det_B = np.linalg.det(B)
    x_M_transpose = np.transpose(X - M)
    x_M = X - M
    inv_B = np.linalg.inv(B)

    return np.log(P) - np.log(np.sqrt(det_B)) - 0.5 * __mul_3_matrices(x_M_transpose, inv_B, x_M)


def get_experimental_probability(p, N):
    return np.sqrt((1 - p) / (N * p))