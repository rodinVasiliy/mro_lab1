import matplotlib.pyplot as plt
import numpy as np


def load_features(filename):
    return np.load(filename)


def show_vector_points(X, color='red'):
    plt.scatter(X[0, :], X[1, :], color=color)


def get_bayesian_border_for_normal_classes(X, M_l, M_j, B, thresh):
    def dividing_border(x, coeff_1, coeff_2):
        return np.float_(np.dot(coeff_1, x) + coeff_2 + thresh)

    Y = []
    M_diff = (M_l - M_j)
    M_diff_T = M_diff.reshape(1, 2)
    M_sum = M_l + M_j
    B_inv = np.linalg.inv(B)
    coeff_1 = np.dot(M_diff_T, B_inv)
    coeff_2 = -0.5 * np.dot(np.dot(M_diff_T, B_inv), M_sum)

    for x in X:
        x_coord = np.array(x, 0).reshape(2, 1)
        Y.append(dividing_border(x_coord, coeff_1, coeff_2))

    return np.array(Y)


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
    x_border = np.linspace(-1, 1, 100)
    y_border = get_bayesian_border_for_normal_classes(x_border, M1, M2, R1, np.log(P2 / P1))
    plt.plot(x_border, y_border, color='green')
    show_vector_points(feature1, 'red')
    show_vector_points(feature2, 'blue')
    plt.show()
