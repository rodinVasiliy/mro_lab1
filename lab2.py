import matplotlib.pyplot as plt
import numpy as np
import lab1
from scipy.stats import norm
from scipy.special import erfinv


def load_features(filename):
    return np.load(filename)


def show_vector_points(X, color='red'):
    plt.scatter(X[0, :], X[1, :], color=color)


def get_bayesian_border_for_normal_classes_with_same_cor_matrix(X, M_l, M_j, B, thresh):
    M_diff = M_l - M_j
    M_diff_T = M_diff.reshape(1, 2)
    M_sum = (M_l + M_j)
    M_sum_T = M_sum.reshape(1, 2)
    B_inv = np.linalg.inv(B)
    b_matrix = np.dot(M_diff_T, B_inv)
    c = 0.5 * np.dot(np.dot(M_sum_T, B_inv), M_diff) - thresh
    Y = []
    for x in X:
        y = (c - b_matrix[0, 0] * x) / b_matrix[0, 1]
        Y.append(np.float_(y))
    return np.array(Y)


def get_bayesian_border_for_normal_classes(X, M_l, M_j, B_l, B_j, thresh):
    bound = []

    B_j_inv = np.linalg.inv(B_j)
    B_l_inv = np.linalg.inv(B_l)
    b_matrix_1 = B_j_inv - B_l_inv
    M_l_T = M_l.reshape(1, 2)
    M_j_T = M_j.reshape(1, 2)
    b_matrix_2 = 2 * (np.dot(M_l_T, B_l_inv) - np.dot(M_j_T, B_j_inv))
    c = np.log(np.linalg.det(B_l) / np.linalg.det(B_j)) + 2 * thresh - np.dot(np.dot(M_l_T, B_l_inv),
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


def get_prob_wrong_classification(M1, M2, R):
    m_d = lab1.get_M_distance(M1, M2, R)
    p0 = 1 - norm.cdf(1 / 2 * np.sqrt(m_d))
    p1 = norm.cdf(-1 / 2 * np.sqrt(m_d))
    return p0, p1


def get_lambda(M_l, M_j, R, p0_s):
    M_distance = lab1.get_M_distance(M_l, M_j, R)
    return np.exp(-0.5 * M_distance + np.sqrt(M_distance) * erfinv(1 - p0_s))


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
    x_array = np.linspace(-2, 1, 100)
    y_array = get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_array, M1, M2, R1, thresh=np.log(P2 / P1))
    plt.plot(x_array, y_array, color='green')
    show_vector_points(feature1, 'red')
    show_vector_points(feature2, 'blue')
    plt.title('bayes')
    plt.show()
    prob_wrong_1, prob_wrong_2 = get_prob_wrong_classification(M1, M2, R1)
    print("Вероятность ошибочной классификации для первого класса: ", prob_wrong_1)
    print("Вероятность ошибочной классификации для второго класса: ", prob_wrong_2)
    sum_wrong_prob = prob_wrong_1 + prob_wrong_2
    print("Сумма точечных оценок вероятностей ошибочной классификации для первого и второго класса: ", sum_wrong_prob)

    # minmax
    P1_optim = 0.5
    P2_optim = 0.5
    x_array = np.linspace(-2, 1, 100)
    y_array = get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_array, M1, M2, R1, thresh=np.log(P2 / P1))
    plt.plot(x_array, y_array, color='green')
    show_vector_points(feature1, 'red')
    show_vector_points(feature2, 'blue')
    plt.title('minmax')
    plt.show()

    # N-P
    p0_s = 0.05
    thresh_lambda = get_lambda(M1, M2, R1, p0_s)
    y_array = get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_array, M1, M2, R1, thresh=thresh_lambda)
    plt.plot(x_array, y_array, color='green')
    show_vector_points(feature1, 'red')
    show_vector_points(feature2, 'blue')
    plt.title('N-P')
    plt.show()

    feature1, feature3, feature4 = load_features('C:\\mro_lab1\\three_classes.npy')
    x_array = np.linspace(-3, 2, 200)
    bound_1_3 = get_bayesian_border_for_normal_classes(x_array, M1, M2, R1, R2, np.log(P2 / P1))
    bound_1_4 = get_bayesian_border_for_normal_classes(x_array, M1, M3, R1, R3, np.log(P2 / P1))
    bound_3_4 = get_bayesian_border_for_normal_classes(x_array, M2, M3, R2, R3, np.log(P2 / P1))
    plt.ylim(-4, 3)
    plt.xlim(-3, 2)
    # передаем координаты x[:, 0] и координаты y[:, 1]
    plt.scatter(bound_1_3[:, 0], bound_1_3[:, 1], color='orange')
    plt.scatter(bound_1_4[:, 0], bound_1_4[:, 1], color='black')
    plt.scatter(bound_3_4[:, 0], bound_3_4[:, 1], color='yellow')
    show_vector_points(feature1, 'red')
    show_vector_points(feature3, 'blue')
    show_vector_points(feature4, 'green')
    plt.title('bayes for dif correlation matrix')
    plt.show()
