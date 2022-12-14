import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

import Constants
import lab1
import lab2
import lab4
import lab6
from lab1 import get_training_samples
from utils.baios import get_bayos_lines

def get_euclidean_distance(x, y):
    x = x.reshape(2, )
    y = y.reshape(2, )
    return np.linalg.norm(x - y)
    # return distance.euclidean(x, y)


def get_distances_array(x, datasets, datasets_labels):
    distances_with_labels = []
    count_labels = 0
    for dataset in datasets:
        for j in range(dataset.shape[-1]):
            distances_with_labels.append((get_euclidean_distance(x, dataset[:, j]), datasets_labels[count_labels]))
            count_labels += 1
    return distances_with_labels


def get_B(dataset):
    return np.cov(dataset)


def get_h(dataset, k):
    n = dataset.shape[0]
    N = dataset.shape[1]
    return pow(N, -k / n)


def get_a_prior_probs(datasets):
    N_values = []
    for dataset in datasets:
        N_values.append(dataset.shape[-1])

    P_values = []
    for dataset in datasets:
        P_values.append(dataset.shape[-1] / np.sum(N_values))

    return np.array(P_values)


def get_f_estimate(x, train_dataset):
    n = train_dataset.shape[0]
    N = train_dataset.shape[1]
    k = Constants.k
    B = get_B(train_dataset)
    inv_B = np.linalg.inv(B)
    det_B = np.linalg.det(B)
    h = get_h(train_dataset, k)
    tmp = 1 / (pow(2 * np.pi, n / 2) * np.sqrt(det_B) * pow(h, n))
    h_ = -0.5 * pow(h, -2)
    sum = 0
    for i in range(0, N):
        xi = train_dataset[:, i]
        x_minus_xi = (x - xi).reshape(1, 2)
        x_minus_xi_T = x_minus_xi.reshape(2, 1)
        sum += tmp * np.exp(h_ * np.matmul(np.matmul(x_minus_xi, inv_B), x_minus_xi_T))
    return sum / N


def K_neighbours_classifier(x, datasets, params):
    K = params[0]
    labels = params[1]
    array_distances = get_distances_array(x, datasets, labels)
    array_distances.sort(key=lambda row: row[0])
    array_distances = np.array(array_distances)
    K_0 = 0
    K_1 = 0
    for i in range(0, K):
        if array_distances[i, 1] == 0:
            K_0 += 1
        else:
            K_1 += 1
    K_array = [K_0, K_1]
    return np.argmax(K_array)


def parzen_classifier(x, datasets, args=0):
    P_values = get_a_prior_probs(datasets)

    density_values = []
    for dataset in datasets:
        density_values.append(get_f_estimate(x, dataset))
    density_values = np.array(density_values)[:, 0]

    result = P_values * density_values
    result = result[:, 0]

    return np.argmax(result)


def classify_data(train_samples, testing_samples, classifier, classifier_param):
    test_results = []
    for test_sample in testing_samples:
        test_result = []

        for j in range(test_sample.shape[-1]):
            test_result.append(classifier(test_sample[:, j], train_samples, classifier_param))

        test_results.append(np.array(test_result))

    return np.array(test_results)


def get_classification_error_from_labels(y_pred, y_test):
    N0 = np.sum(y_test == 0)
    N1 = np.sum(y_test == 1)
    y_pred = np.ravel(y_pred)
    p0_count = 0
    p1_count = 0
    for test_label, pred_label in zip(y_test, y_pred):
        if test_label == 0 and pred_label == 1:
            p0_count += 1
        if test_label == 1 and pred_label == 0:
            p1_count += 1
    return p0_count / N0, p1_count / N1


def get_erorrs_indexes(y_pred, y_test):
    y_pred = np.ravel(y_pred)
    erorrs_indexes = []
    for i in range(0, len(y_pred)):
        if int(y_test[i]) != int(y_pred[i]):
            erorrs_indexes.append(i)
    return np.array(erorrs_indexes)


def show_wrong_classification(samples, indexes):
    for index in indexes:
        plt.scatter(samples[0, index], samples[1, index], marker='o', color='black', alpha=0.4, s=100)


if __name__ == '__main__':
    M0p = Constants.M1_lab5
    M1p = Constants.M2_lab5
    b0p = Constants.R1_lab5
    b1p = Constants.R2_lab5
    params_array = [[M0p, M1p, b0p, b0p], [M0p, M1p, b0p, b1p]]
    case_array = ['???????????? ??????????????-???????????????????? ??????????????', '???????????? ??????????????-???????????????????????? ??????????????']
    for params, case in zip(params_array, case_array):
        print(case)
        M0 = params[0]
        M1 = params[1]
        b0 = params[2]
        b1 = params[3]
        train_dataset0 = get_training_samples(M0, b0, 50)
        train_dataset1 = get_training_samples(M1, b1, 50)
        plt.show()
        train_datasets = [train_dataset0, train_dataset1]
        test_datasets = [lab1.get_training_samples(M0, b0, 100), get_training_samples(M1, b1, 100)]

        print("\n?????????????????????????? ????????????, 2 ????????????")
        p0 = lab2.classification_errordiff(test_datasets[0], M0, M1, b0, b1, 0.5, 0.5)
        p1 = lab2.classification_errordiff(test_datasets[1], M1, M0, b1, b0, 0.5, 0.5)
        print(f"?????????????????????? ?????????????????? ?????????????????????????? p01: {p0}")
        print(f"?????????????????????? ?????????????????? ?????????????????????????? p10: {p1}")
        print(f"???????????????????????? ????????: {0.5 * p0 + 0.5 * p1}")

        y_test = np.zeros(100)
        y_test = np.concatenate((y_test, np.ones(100)), axis=0)
        y_pred = classify_data(train_datasets, test_datasets, parzen_classifier, None)
        error_indexes = get_erorrs_indexes(y_pred, y_test)
        plt.title('show classification parzen')
        lab1.show_vector_points1(test_datasets[0], 'red')
        lab1.show_vector_points1(test_datasets[1], 'blue')
        show_wrong_classification(np.concatenate((test_datasets[0], test_datasets[1]), axis=1), error_indexes)
        x_array = np.arange(-1, 4, 0.01)
        thresh = np.log(0.5 / 0.5)
        if case == '???????????? ??????????????-???????????????????? ??????????????':
            bayes_ = lab2.get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_array, M0, M1,
                                                                                      b0, thresh)
            plt.plot(x_array, bayes_, color='green')
        else:
            x_array = np.arange(-1, 4, 0.01)
            bayes_ = lab2.get_bayesian_border_for_normal_classes(x_array, M0, M1, b0, b1, thresh)
            lab6.show_bayes_border(bayes_, 'green', 'bs', '.')
        plt.ylim([-2.5, 4])
        plt.show()
        print(error_indexes)
        p0, p1 = get_classification_error_from_labels(y_pred, y_test)

        print("\n?????????? ??????????????, 2 ????????????")
        print(f"?????????????????????? ?????????????????? ?????????????????????????? p01: {p0}")
        print(f"?????????????????????? ?????????????????? ?????????????????????????? p10: {p1}")
        print(f"???????????????????????? ????????: {0.5 * p0 + 0.5 * p1}")

        labels = np.zeros(50)
        labels = np.concatenate((labels, np.ones(50)), axis=0)
        y_test = np.zeros(100)
        y_test = np.concatenate((y_test, np.ones(100)), axis=0)
        K_array = np.arange(1, 7, 2)
        for K in K_array:
            params = [K, labels]
            y_pred = classify_data(train_datasets, test_datasets, K_neighbours_classifier, params)
            error_indexes = get_erorrs_indexes(y_pred, y_test)
            print(error_indexes)
            plt.title(f'show classification K = {K}')
            lab1.show_vector_points1(test_datasets[0], 'red')
            lab1.show_vector_points1(test_datasets[1], 'blue')
            show_wrong_classification(np.concatenate((test_datasets[0], test_datasets[1]), axis=1), error_indexes)
            if case == '???????????? ??????????????-???????????????????? ??????????????':
                bayes_ = lab2.get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_array, M0, M1,
                                                                                          b0, thresh)
                plt.plot(x_array, bayes_, color='green')
            else:
                x_array = np.arange(-1, 4, 0.01)
                bayes_ = lab2.get_bayesian_border_for_normal_classes(x_array, M0, M1, b0, b1, thresh)
                lab6.show_bayes_border(bayes_, 'green', 'bs', '.')
            plt.ylim([-2.5, 4])
            plt.show()

            p0, p1 = get_classification_error_from_labels(y_pred, y_test)
            print(f"\n?????????? K ?????????????????? (K = {K}) ?????????????? 2 ????????????")
            print(f"?????????????????????? ?????????????????? ?????????????????????????? p01: {p0}")
            print(f"?????????????????????? ?????????????????? ?????????????????????????? p10: {p1}")
            print(f"???????????????????????? ????????: {0.5 * p0 + 0.5 * p1}")
