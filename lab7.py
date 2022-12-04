import lab1
import lab2
import lab6
from lab1 import get_training_samples
import numpy as np
import matplotlib.pyplot as plt
from lab5 import get_euclidean_distance


def show_samples(samples_array, colors_array):
    for samples, colors in zip(samples_array, colors_array):
        plt.scatter(samples[0, :], samples[1, :], color=colors)


def concatenate_samples(samples_array):
    result = samples_array[0]
    for i in range(1, len(samples_array)):
        result = np.concatenate((result, samples_array[i]), axis=1)
    return result


def find_first_center_and_max_distance(samples):
    samples_mean = np.mean(samples, axis=1)
    samples_mean = samples_mean.reshape(2, 1)
    distances = []
    for i in range(0, len(samples[1])):
        distances.append(get_euclidean_distance(samples_mean, samples[0:2, i]))
    m0 = np.argmax(distances)
    return samples[0:2, m0], distances[m0]


def find_second_center_and_max_distance(samples, m0):
    distances = []
    for i in range(0, len(samples[1])):
        distances.append(get_euclidean_distance(m0, samples[0:2, i]))
    m1 = np.argmax(distances)
    return samples[0:2, m1], distances[m1]


def remove_center_from_samples(samples, center):
    center = center.reshape(2, 1)
    copy = samples.copy()
    del_index = np.where(copy[0:2, :] == center)
    return np.delete(copy, del_index[1][1], axis=1)


def get_distances_to_centers(samples, centers_array):
    N = len(samples[1])
    distances = np.zeros(shape=(len(centers_array), N))
    for i in range(0, len(centers_array)):
        center = centers_array[i]
        for j in range(0, N):
            distances[i, j] = get_euclidean_distance(center, samples[0:2, j])
    return distances


def get_min_distances(min_indexes, distances_array):
    res = []
    for i in range(0, len(min_indexes)):
        res.append(distances_array[min_indexes[i], i])
    return np.array(res)


def get_sum_distance(distances_array):
    distance = 0
    for distances in distances_array:
        distance += sum(distances)
    return distance


def get_typical_distance(centers_array):
    center_distances = []
    i = 0
    for center in centers_array:
        center = center.reshape(2, 1)
        center_distances.append(get_distances_to_centers(center, centers_array))
        i += 1
    sum_distance = get_sum_distance(np.triu(center_distances))
    return 0.5 * sum_distance / len(center_distances)


def get_centres(samples):
    m0, m0_max_dist = find_first_center_and_max_distance(samples)
    m1, m1_max_dist = find_second_center_and_max_distance(samples, m0)
    centers_array = [m0, m1]
    remaining_samples = remove_center_from_samples(samples, m0)
    remaining_samples = remove_center_from_samples(remaining_samples, m1)
    max_distance_array = [m0_max_dist, m1_max_dist]
    t_distance_array = [0, 0]
    while True:
        distances_array = get_distances_to_centers(remaining_samples, centers_array)
        min_indexes = np.argmin(distances_array, axis=0)
        distances_array_min = get_min_distances(min_indexes, distances_array)
        max_indexes = np.argmax(distances_array_min)
        max_distance_array.append(distances_array_min[max_indexes])
        centers_candidate = remaining_samples[0:2, max_indexes]
        typical_distance = get_typical_distance(np.array(centers_array))
        t_distance_array.append(typical_distance)
        tmp_center_candidate = centers_candidate.reshape(2, 1)
        distances_array_from_centers = get_distances_to_centers(tmp_center_candidate, centers_array)
        d_min_index = np.argmin(distances_array_from_centers)
        d_min = distances_array_from_centers[d_min_index]
        if d_min < typical_distance:
            break
        centers_array.append(centers_candidate)
        remaining_samples = remove_center_from_samples(samples, centers_candidate)
    return centers_array, max_distance_array, t_distance_array


if __name__ == '__main__':
    N = 50
    M1 = np.array([1, -1]).reshape(2, 1)
    M2 = np.array([2, 2]).reshape(2, 1)
    M3 = np.array([-1, 1]).reshape(2, 1)
    M4 = np.array([1, 1]).reshape(2, 1)
    M5 = np.array([-1, -1]).reshape(2, 1)
    M = [M1, M2, M3, M4, M5]
    B0 = np.array([[0.03, 0.01], [0.01, 0.02]])
    B1 = np.array([[0.02, -0.01], [-0.01, 0.03]])
    B2 = np.array([[0.015, 0.015], [0.015, 0.02]])
    B3 = np.array([[0.04, 0.0], [0.0, 0.03]])
    B4 = np.array([[0.01, -0.01], [-0.01, 0.015]])
    B = [B0, B1, B2, B3, B4]
    # samples1 = get_training_samples(M[0], B[0], 50)
    # samples2 = get_training_samples(M[1], B[1], 50)
    # samples3 = get_training_samples(M[2], B[2], 50)
    # samples4 = get_training_samples(M[3], B[3], 50)
    # samples5 = get_training_samples(M[4], B[4], 50)
    samples1, samples2, samples3, samples4, samples5 = lab2.load_features('five_classes.npy')
    colors_array = ['red', 'green', 'blue', 'pink', 'yellow']
    show_samples([samples1, samples2, samples3, samples4, samples5], colors_array)
    plt.show()

    samples_array = [samples1, samples2, samples3, samples4, samples5]
    samples_array_result = concatenate_samples(samples_array)
    for i in range(2, 6):
        fig = plt.figure(figsize=(15, 5))
        fig.add_subplot(1, 2, 1)
        plt.title(f'minmax for {i} classes')
        res = samples_array_result[:, 0:N * i]
        m_array, max_dist_array, t_dist_array = get_centres(res)
        print(m_array)
        for j in range(0, i):
            lab1.show_vector_points1(samples_array[j], colors_array[j])
        for m in m_array:
            plt.scatter(m[0], m[1], marker='o', color='black', alpha=0.6, s=100)

        fig.add_subplot(1, 2, 2)
        plt.title(f'minmax for {i} classes')
        x = np.arange(0, len(m_array) + 1)
        plt.plot(x, max_dist_array, label='max distance')
        plt.plot(x, t_dist_array, label='typical distance')
        plt.legend()
        plt.show()
