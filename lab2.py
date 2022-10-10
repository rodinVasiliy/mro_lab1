import matplotlib.pyplot as plt
import numpy as np


def load_features(filename):
    return np.load(filename)


def show_vector_points(X, color='red'):
    plt.scatter(X[0, :], X[1, :], color=color)


def get_bayesian_border_for_normal_classes_with_same_cor_matrix(X, M_l, M_j, B, P_l, P_j):
    M_diff = M_l - M_j
    M_diff_T = M_diff.reshape(1, 2)
    M_sum = (M_l + M_j)
    M_sum_T = M_sum.reshape(1, 2)
    B_inv = np.linalg.inv(B)
    b_matrix = np.dot(M_diff_T, B_inv)
    c = 0.5 * np.dot(np.dot(M_sum_T, B_inv), M_diff) - np.log(P_l / P_j)
    Y = []
    for x in X:
        y = (c - b_matrix[0, 0] * x) / b_matrix[0, 1]
        Y.append(np.float_(y))
    return np.array(Y)


def get_bayesian_border_for_normal_classes(X, M_l, M_j, B_l, B_j, P_l, P_j):
    bound = []

    B_j_inv = np.linalg.inv(B_j)
    B_l_inv = np.linalg.inv(B_l)
    b_matrix_1 = B_j_inv - B_l_inv
    M_l_T = M_l.reshape(1, 2)
    M_j_T = M_j.reshape(1, 2)
    b_matrix_2 = 2 * (np.dot(M_l_T, B_l_inv) - np.dot(M_j_T, B_j_inv))
    c = np.log(np.linalg.det(B_l) / np.linalg.det(B_j)) + 2 * np.log(P_l / P_j) - np.dot(np.dot(M_l_T, B_l_inv),
                                                                                         M_l) + np.dot(
        np.dot(M_j_T, B_j_inv), M_j)
    A = b_matrix_1[1, 1]

    for x in X:
        B = (b_matrix_1[0, 1] + b_matrix_1[1, 0]) * x + b_matrix_2[0, 1]
        C = b_matrix_1[0, 0] * (x ** 2) + b_matrix_2[0, 0] * x + c
        D = (B ** 2) - 4 * A * C
        if D >= 0:
            y1 = ((-1) * B + np.sqrt(D)) / (2 * A)
            y2 = ((-1) * B - np.sqrt(D)) / (2 * A)
            if y1 == y2:
                bound += [x, y1.item()]
            else:
                bound += [[x, y1.item()], [x, y2.item()]]
    return np.array(bound)


if __name__ == '__main__':
    M1 = np.array([-1, 1]).reshape(2, 1)
    M2 = np.array([0, 1]).reshape(2, 1)
    M3 = np.array([-1, -1]).reshape(2, 1)
    R1 = np.array(([0.08, 0.05], [0.05, 0.05]))
    R2 = np.array(([0.1, 0], [0, 0.1]))
    R3 = np.array(([0.87, -0.87], [-0.87, 0.95]))
    P1 = 0.5
    P2 = 0.5
    P3 = 0.5
    feature1, feature2 = load_features('C:\\mro_lab1\\two_classes.npy')
    x_border = np.linspace(-2, 1, 100)
    y_border = get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_border, M1, M2, R1, P1, P2)
    plt.plot(x_border, y_border, color='green')
    show_vector_points(feature1, 'red')
    show_vector_points(feature2, 'blue')
    plt.show()
    feature1, feature3, feature4 = load_features('C:\\mro_lab1\\three_classes.npy')
    x_border = np.linspace(-3, 2, 200)
    bound_1_3 = get_bayesian_border_for_normal_classes(x_border, M1, M2, R1, R2, P1, P2)
    bound_1_4 = get_bayesian_border_for_normal_classes(x_border, M1, M3, R1, R3, P1, P2)
    bound_3_4 = get_bayesian_border_for_normal_classes(x_border, M2, M3, R2, R3, P1, P2)
    plt.ylim(-4, 3)
    plt.xlim(-3, 2)
    plt.scatter(bound_1_3[:, 0], bound_1_3[:, 1], color='orange')
    plt.scatter(bound_1_4[:, 0], bound_1_4[:, 1], color='black')
    plt.scatter(bound_3_4[:, 0], bound_3_4[:, 1], color='yellow')
    show_vector_points(feature1, 'red')
    show_vector_points(feature3, 'blue')
    show_vector_points(feature4, 'green')
    plt.show()
