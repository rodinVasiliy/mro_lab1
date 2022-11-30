import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show

import Constants
import lab1
import lab2
from lab1 import get_training_samples, save_generated_features, show_vector_points1


def get_a_prior_probs(datasets):
    N_values = []
    for dataset in datasets:
        N_values.append(dataset.shape[-1])

    P_values = []
    for dataset in datasets:
        P_values.append(dataset.shape[-1] / np.sum(N_values))

    return np.array(P_values)


def parzen_classifier(x, datasets, args=0):
    P_values = get_a_prior_probs(datasets)

    density_values = []
    for dataset in datasets:
        density_values.append(Kernel_Density_Estimation_Normal(x, dataset))
    density_values = np.array(density_values)[:, 0]

    result = P_values * density_values

    return np.argmax(result)


def classify_data(train_samples, testing_samples, classifier, classifier_param):
    test_results = []
    for test_sample in testing_samples:
        test_result = []

        for j in range(test_sample.shape[-1]):
            test_result.append(classifier(test_sample[:, :, j], train_datasets, classifier_param))

        test_results.append(np.array(test_result))

    return np.array(test_results)


if __name__ == '__main__':
    M0 = Constants.M1
    M1 = Constants.M2
    b = Constants.R1

    train_datasets = [lab1.get_training_samples(M0, b, 50), get_training_samples(M1, b, 50)]
    test_datasets = [lab1.get_training_samples(M0, b, 100), get_training_samples(M1, b, 100)]

    print("\nклассификатор Байеса, равные корреляционные матрицы, 2 класса")
    p0 = lab2.classification_errordiff(test_datasets[0], M0, M1, b, b, 0.5, 0.5)
    p1 = lab2.classification_errordiff(test_datasets[1], M1, M0, b, b, 0.5, 0.5)
    print(f"Вероятность ошибочной классификации p01: {p0}")
    print(f"Вероятность ошибочной классификации p10: {p1}")
    print(f"Эмпирический риск: {0.5 * p0 + 0.5 * p1}")
