import random

import numpy as np
from matplotlib import pyplot as plt
from qpsolvers import solve_qp
from sklearn import svm

import Constants
import lab1
import lab4
from sklearn.svm import SVC, LinearSVC
import utils.show


def get_svc(C, K, K_params):
    if K == 'poly':
        return SVC(C=C, kernel=K, degree=K_params[1], coef0=K_params[0])
    if K == 'rbf':
        return SVC(C=C, kernel=K, gamma=K_params[0])  # radial
    if K == 'sigmoid':
        return SVC(C=C, kernel=K, coef0=K_params[1], gamma=K_params[0])


def get_discriminant_kernel(support_vectors, lambda_r, x, K, K_params):
    sum = 0
    for j in range(support_vectors.shape[1]):
        sum += lambda_r[j] * get_K(support_vectors[0:2, j].reshape(2, 1), x, K, K_params)
    return sum


def get_K(x, y, K, K_params):
    if K == 'poly':
        c = K_params[0]
        d = K_params[1]
        if x.shape != (1, 2):
            x = x.reshape(1, 2)
        tmp = np.matmul(x, y) + c
        return pow(tmp, d)
    if K == 'rbf':
        gamma = K_params[0]
        return np.exp(-gamma * np.sum(np.power((x - y), 2)))
    if K == 'sigmoid':
        if x.shape != (1, 2):
            x = x.reshape(1, 2)
        gamma = K_params[0]
        c = K_params[1]
        return np.tanh(gamma * np.matmul(x, y)[0] + c)
    return None


def get_P_kernel(dataset, K, K_params):
    N = dataset.shape[1]
    P = np.ndarray(shape=(N, N))
    for i in range(0, N):
        for j in range(0, N):
            P[i, j] = dataset[2, j] * dataset[2, i] * get_K(dataset[0:2, j], dataset[0:2, i], K, K_params)
    return P


def show_separating_hyperplane(title, w, wn, samples0, samples1, sup_0, sup_1):
    x_range = Constants.x_range_lab6
    x, y = lab4.get_border_lin_classificator(w, wn, x_range)
    utils.show.show(title, samples0, samples1, [x, x + 1 / w[0], x - 1 / w[0]], [y, y, y],
                    ['black', 'green', 'red'], ['', '', ''])
    if sup_0 is not None and sup_0.size > 0:
        plt.scatter(sup_0[0, :], sup_0[1, :], marker='o', color='c', alpha=0.6)
    if sup_1 is not None and sup_1.size > 0:
        plt.scatter(sup_1[0, :], sup_1[1, :], marker='o', color='fuchsia', alpha=0.6)
    plt.show()


def separate_sup_vectors(vectors):
    class_0_vectors = []
    class_1_vectors = []
    tmp = np.transpose(vectors)
    for vec in tmp:
        if vec[2] == -1.0:
            class_0_vectors.append(vec[0:2])
        else:
            class_1_vectors.append(vec[0:2])
    return np.transpose(class_0_vectors), np.transpose(class_1_vectors)


# даны вектора без меток, надо найти
def separate_sup_vectors_with_indexes(dataset, indexes):
    class_0_vectors = []
    class_1_vectors = []
    for index in indexes:
        vect = dataset[:, index]
        if vect[2] == -1.0:
            class_0_vectors.append(vect[0:2])
        else:
            class_1_vectors.append(vect[0:2])
    return np.transpose(class_0_vectors), np.transpose(class_1_vectors)


def get_p_matrix(dataset):
    matrix_range = int(dataset.shape[1])
    new_shape = (matrix_range, matrix_range)
    new_P_matrix = np.zeros(new_shape)
    for i in range(0, matrix_range):
        for j in range(0, matrix_range):
            tmp1 = dataset[2, i] * dataset[2, j]  # zi * zj
            tmp2 = dataset[0, j] * dataset[0, i] + dataset[1, j] * dataset[1, i]
            new_P_matrix[i, j] = float(tmp1 * tmp2)
    return new_P_matrix


def get_a_matrix(dataset):
    a_shape = dataset.shape[1]
    new_a_matrix = np.zeros(a_shape)
    for i in range(0, a_shape):
        new_a_matrix[i] = dataset[2, i]
    return new_a_matrix


def get_q(dataset):
    q_shape = dataset.shape[1]
    return -1 * np.ones(q_shape)


def get_dataset(samples0, samples1):
    new_shape = (samples0.shape[0] + 1, samples0.shape[1] * 2)
    new_dataset = np.zeros(new_shape)
    cnt = 0
    i = 0
    while i != new_dataset.shape[1]:
        new_dataset[0, i] = samples0[0, cnt]
        new_dataset[1, i] = samples0[1, cnt]
        new_dataset[2, i] = -1
        i += 1
        new_dataset[0, i] = samples1[0, cnt]
        new_dataset[1, i] = samples1[1, cnt]
        new_dataset[2, i] = 1
        i += 1
        cnt += 1
    return new_dataset


def get_support_vectors(dataset, lambda_array):
    support_vectors = []
    sup_lambda_array = []
    epsilon = 1e-04
    for i in range(0, len(lambda_array)):
        if lambda_array[i] > epsilon:
            support_vectors.append(dataset[:, i])
            sup_lambda_array.append(lambda_array[i])
    return np.array(support_vectors), np.array(sup_lambda_array)


def get_w(support_vectors, lambda_array):
    result = np.zeros(shape=(support_vectors.shape[0] - 1, 1))
    for i in range(0, len(lambda_array)):
        tmp = lambda_array[i] * support_vectors[2, i]
        result[0] += support_vectors[0, i] * tmp
        result[1] += support_vectors[1, i] * tmp
    return result


def get_discriminant_function_for_x(x, sup_vectors, sup_lambdas, K, K_params, wn):
    d_x = 0
    for j in range(0, len(sup_vectors)):
        x_j = sup_vectors[0:2, j]
        lambda_j = sup_lambdas[j]
        r_j = sup_vectors[2, j]
        K_ = get_K(x_j, x, K, K_params)
        d_x += lambda_j * r_j * K_
    return d_x + wn


def get_wn_with_kernel(support_vectors, lambda_array, K, K_params):
    # result = 0
    # for j in range(0, len(support_vectors)):
    #     r_j = support_vectors[2, j]
    #     x_j_sup = support_vectors[0:2, j]
    #     tmp_sum = 0
    #     for i in range(0, len(lambda_array)):
    #         x_i = support_vectors[0:2, i]
    #         lambda_i = lambda_array[i]
    #         r_i = support_vectors[2, i]
    #         tmp_sum += lambda_i * r_i * get_K(x_i, x_j_sup, K, K_params)
    #     result += r_j - tmp_sum
    # return result / len(support_vectors)
    r_j = support_vectors[2, 0]
    y = support_vectors[0:2, 0]
    tmp = 0
    for i in range(0, len(support_vectors)):
        tmp += lambda_array[i] * support_vectors[2, i] * get_K(dataset[0:2, i], y, K, K_params)
    return r_j - tmp


def get_wn(support_vectors, w):
    wn = 0
    w = np.transpose(w)
    for i in range(0, len(support_vectors)):
        r = support_vectors[2, i]
        xi = support_vectors[0:2, i].reshape(2, 1)
        tmp = r - np.matmul(w, xi)
        wn += tmp
    return wn / len(support_vectors)


def get_errors(dataset, w, wn):
    count_errors = 0
    w = np.transpose(w)
    for i in range(0, dataset.shape[1]):
        xi = dataset[0:2, i]
        if np.sign(np.matmul(w, xi) + wn) != np.sign(dataset[2, i]):
            count_errors += 1
    return count_errors / dataset.shape[1]


def find_best_C_from_values(dataset, values):
    errors_array = []
    for C in values:
        clf_svc = svm.SVC(kernel="linear", C=C)
        x = np.transpose(dataset[0:2, :])
        y = dataset[2, :]
        clf_svc.fit(x, y)
        w_svc = clf_svc.coef_.T
        wn_svc = clf_svc.intercept_[0]
        errors = get_errors(dataset, w_svc, wn_svc)
        errors_array.append(errors)
    min_index = np.where(errors_array == np.min(errors_array))
    best_C_value = values[min_index]
    return best_C_value


def get_errors_with_kernel(dataset, sup_vectors, sup_lambdas, K, K_params):
    count_errors = 0
    dataset_len = dataset.shape[1]
    wn = get_wn_with_kernel(sup_vectors, sup_lambdas, K, K_params)
    for j in range(0, dataset_len):
        x_j = dataset[0:2, j]
        r_j = dataset[2, j]
        d_x = get_discriminant_function_for_x(x_j, sup_vectors, sup_lambdas, K, K_params, wn)
        if np.sign(d_x) != np.sign(r_j):
            count_errors += 1
    return count_errors / dataset_len


if __name__ == '__main__':
    # task 1
    # samples0 = lab1.get_training_samples(Constants.M1_lab6, Constants.R1_lab6, 100)
    # samples1 = lab1.get_training_samples(Constants.M2_lab6, Constants.R2_lab6, 100)
    #
    # lab1.show_vector_points1(samples0)
    # lab1.show_vector_points1(samples1, color='blue')
    # plt.show()
    #
    # # task 2
    # dataset = get_dataset(samples0, samples1)
    # p_matrix = get_p_matrix(dataset)
    # a_matrix = get_a_matrix(dataset)
    # q = get_q(dataset)
    # b = np.zeros(1)
    # G = np.eye(dataset.shape[1]) * -1
    # h = np.zeros((dataset.shape[1],))
    # lambda_array = solve_qp(p_matrix, q, G, h, a_matrix, b, solver='cvxopt')
    #
    # sup_vectors, sup_lambdas = get_support_vectors(dataset, lambda_array)
    # sup_vectors = np.transpose(sup_vectors)
    # sup_0, sup_1 = separate_sup_vectors(sup_vectors)
    # w_qp = get_w(sup_vectors, sup_lambdas)
    # wn_qp = get_wn(sup_vectors, w_qp)
    # errors = get_errors(dataset, w_qp, wn_qp)
    # print(f'p0 + p1 errors for qpsolver = {errors}')
    #
    # show_separating_hyperplane('qp', w_qp, wn_qp, samples0, samples1, sup_0, sup_1)
    #
    # clf_svc = svm.SVC(kernel="linear", C=1000)
    # x = np.transpose(dataset[0:2, :])
    # y = dataset[2, :]
    # clf_svc.fit(x, y)
    #
    # support_vectors_svc = clf_svc.support_vectors_
    # support_vectors_svc_indices = clf_svc.support_
    # w_svc = clf_svc.coef_.T
    # wn_svc = clf_svc.intercept_[0]
    # sup_0, sup_1 = separate_sup_vectors_with_indexes(dataset, support_vectors_svc_indices)
    # errors = get_errors(dataset, w_svc, wn_svc)
    # print(f'p0 + p1 errors for SVC = {errors}')
    # show_separating_hyperplane('SVC', w_svc, wn_svc, samples0, samples1, sup_0, sup_1)
    #
    # clf_lin = svm.LinearSVC()
    # clf_lin.fit(x, y)
    # w_linear = clf_lin.coef_.T
    # wn_linear = clf_lin.intercept_[0]
    # errors = get_errors(dataset, w_svc, wn_svc)
    # print(f'p0 + p1 errors for LinearSVC = {errors}')
    # show_separating_hyperplane('LinearSVC', w_linear, wn_linear, samples0, samples1, None, None)

    # task 3
    # samples0 = lab1.get_training_samples(Constants.M1_lab6, Constants.R1_lab6, 100)
    # samples1 = lab1.get_training_samples(Constants.M2_lab6_, Constants.R2_lab6, 100)
    #
    # # lab1.show_vector_points1(samples0)
    # # lab1.show_vector_points1(samples1, color='blue')
    # # plt.show()
    #
    # dataset = get_dataset(samples0, samples1)
    # dataset_len = dataset.shape[1]
    # p_matrix = get_p_matrix(dataset)
    # a_matrix = get_a_matrix(dataset)
    # q = get_q(dataset)
    # b = np.zeros(1)
    # G = np.concatenate((np.eye(dataset_len) * -1, np.eye(dataset_len)), axis=0)
    # x = np.transpose(dataset[0:2, :])
    # y = dataset[2, :]
    # for C in [0.1, 1, 10]:
    #     h = np.concatenate((np.zeros((dataset_len,)), np.full((dataset_len,), C)), axis=0)
    #
    #     lambda_array = solve_qp(p_matrix, q, G, h, a_matrix, b, solver='cvxopt')
    #
    #     sup_vectors, sup_lambdas = get_support_vectors(dataset, lambda_array)
    #     sup_vectors = np.transpose(sup_vectors)
    #     sup_0, sup_1 = separate_sup_vectors(sup_vectors)
    #     w_qp = get_w(sup_vectors, sup_lambdas)
    #     wn_qp = get_wn(sup_vectors, w_qp)
    #     errors = get_errors(dataset, w_qp, wn_qp)
    #     print(f'C = {C} p0 + p1 errors for qpsolver = {errors}')
    #
    #     show_separating_hyperplane(f'C = {C} qp', w_qp, wn_qp, samples0, samples1, sup_0, sup_1)
    #
    #     clf_svc = svm.SVC(kernel="linear", C=C)
    #
    #     clf_svc.fit(x, y)
    #
    #     support_vectors_svc = clf_svc.support_vectors_
    #     w_svc = clf_svc.coef_.T
    #     wn_svc = clf_svc.intercept_[0]
    #     sup_0, sup_1 = separate_sup_vectors(support_vectors_svc)
    #     errors = get_errors(dataset, w_svc, wn_svc)
    #     print(f'C = {C} p0 + p1 errors for SVC = {errors}')
    #     show_separating_hyperplane(f'C = {C} SVC method', w_svc, wn_svc, samples0, samples1, sup_0, sup_1)
    #
    # print(f'best C = {find_best_C_from_values(dataset, np.arange(1, 20, 1))}')

    # task 4
    samples0 = lab1.get_training_samples(Constants.M1_lab6, Constants.R1_lab6, 100)
    samples1 = lab1.get_training_samples(Constants.M2_lab6_, Constants.R2_lab6, 100)
    dataset = get_dataset(samples0, samples1)
    dataset_len = dataset.shape[1]

    # K = 'poly'
    # K_params = [0, 3]

    K = 'rbf'
    K_params = [1]

    # K = 'rbf'
    # var = np.var(np.sqrt(np.power(dataset[0:2, :], 2) + np.power(dataset[0:2, :], 2)))
    # K_params = [1 / (2 * var)]

    # K = 'sigmoid'
    # K_params = [1 / 14, -1]

    P = get_P_kernel(dataset, K, K_params)
    a_matrix = get_a_matrix(dataset)
    q = np.full((dataset_len, 1), -1, dtype=np.double)
    b = np.zeros(1)
    G = np.concatenate((np.eye(dataset_len) * -1, np.eye(dataset_len)), axis=0)

    X = np.transpose(dataset[0:2, :])
    Y = dataset[2, :]
    eps = 1e-04
    for C in [0.1, 1, 10, 20]:
        h = np.concatenate((np.zeros((dataset_len,)), np.full((dataset_len,), C)), axis=0)
        lambda_array = solve_qp(P, q, G, h, a_matrix, b, solver='cvxopt')
        support_vectors_positions = lambda_array > eps
        sup_vectors, sup_lambdas = get_support_vectors(dataset, lambda_array)
        sup_vectors = np.transpose(sup_vectors)
        sup_0, sup_1 = separate_sup_vectors(sup_vectors)

        r_vectors = sup_vectors[2, :]
        w_N = []
        for j in range(sup_vectors.shape[1]):
            w_N.append(get_discriminant_kernel(sup_vectors, (lambda_array * a_matrix)[support_vectors_positions],
                                               sup_vectors[0:2, j].reshape(2, 1), K, K_params))
        w_N = np.mean(r_vectors - np.array(w_N))

        cnt_errors = get_errors_with_kernel(dataset, sup_vectors, sup_lambdas, K, K_params)
        print(f'C = {C}, method is qp, count errors for K = {K} = {cnt_errors}')

        y = np.linspace(-4, 4, dataset_len)
        x = np.linspace(-4, 4, dataset_len)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T

        discriminant_func_values = []
        for i in range(xy.shape[0]):
            discriminant_func_values.append(get_discriminant_kernel(sup_vectors,
                                                                    (lambda_array * a_matrix)[
                                                                        support_vectors_positions],
                                                                    xy[i].reshape(2, 1), K, K_params)
                                            + w_N)
        discriminant_func_values = np.array(discriminant_func_values).reshape(xx.shape)

        utils.show.show(f"solve_qp ({K}) C={C}", samples0, samples1, [], [], ['black', 'green', 'red'],
                        ['', '', ''])
        plt.contour(xx, yy, discriminant_func_values, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        if sup_0 is not None and sup_0.size > 0:
            plt.scatter(sup_0[0, :], sup_0[1, :], marker='o', color='c', alpha=0.6, edgecolors='black')
        if sup_1 is not None and sup_1.size > 0:
            plt.scatter(sup_1[0, :], sup_1[1, :], marker='o', color='fuchsia', alpha=0.6)
        plt.show()

        clf = get_svc(C, K, K_params)

        clf.fit(X, Y)
        support_vectors_svc = clf.support_vectors_
        support_vectors_indexes = clf.support_
        sup_0, sup_1 = separate_sup_vectors_with_indexes(dataset, support_vectors_indexes)

        y = np.linspace(-4, 4, dataset_len)
        x = np.linspace(-4, 4, dataset_len)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T
        discriminant_func_values_svc = clf.decision_function(xy).reshape(xx.shape)

        utils.show.show(f"SVC ({K}) C={C}", samples0, samples1, [], [], ['black', 'green', 'red'],
                        ['', '', ''])
        plt.contour(xx, yy, discriminant_func_values_svc, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        if sup_0 is not None and sup_0.size > 0:
            plt.scatter(sup_0[0, :], sup_0[1, :], marker='o', color='c', alpha=0.6)
        if sup_1 is not None and sup_1.size > 0:
            plt.scatter(sup_1[0, :], sup_1[1, :], marker='o', color='fuchsia', alpha=0.6)
        plt.show()
