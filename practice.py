import numpy as np
from matplotlib import pyplot as plt


from lab7 import k_means_method

samples = np.array(([0, 0], [0, 1], [5, 4], [5, 5], [4, 5], [1, 0]))

if __name__ == '__main__':
    samples = samples.T
    K = 2
    rng = np.random.default_rng(42)
    indexes = rng.choice(range(samples.shape[1]), K, replace=False)
    indexes[0] = 0
    indexes[1] = 1
    centers, classes, stats = k_means_method(samples, K, indexes)
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    plt.title(f'k means for {K} classes')
    for k in range(len(classes)):
        plt.scatter(classes[k][:, 0], classes[k][:, 1], label=f"cl{k}")
    for c in centers:
        plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)
    fig.add_subplot(1, 2, 2)
    plt.title(f'k means for {K} classes')
    x = np.arange(3, 3 + len(stats))
    plt.plot(x, stats, label='dependence of the number of changes on the iteration number')
    plt.xlabel('count iteration')
    plt.ylabel('count changed vectors')
    plt.legend()
    plt.show()
