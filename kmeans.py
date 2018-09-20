# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.cluster.vq import kmeans, vq

COLORS = ['r', 'g', 'c', 'm', 'y', 'k']

def load_dataset(file):
    return np.loadtxt(file)

def euclidian(a, b):
    return np.linalg.norm(a-b)

def plot_process(dataset, history_centroids, belongs_to):
    fig, ax = plt.subplots()

    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (COLORS[index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                plt.pause(0.8)

def plot(dataset, centroids, belongs_to):
    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            plt.plot(dataset[instance_index][0], dataset[instance_index][1], (COLORS[index] + 'o'))

    for x, y in centroids:
        plt.plot(x, y, 'bo')
    plt.show()


def kmeans_(dataset, k, epsilon=0, distance="euclidian"):
    history_centroids = []
    if distance == "euclidian":
        dist_method = euclidian
    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k,1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype, instance)
            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes
        history_centroids.append(prototypes)

    return prototypes, history_centroids, belongs_to

def main():
    dataset = load_dataset("durudataset.txt")
    centroids, history_centroids, belongs_to = kmeans_(dataset, 4)
    # plot(dataset, history_centroids, belongs_to)
    centroids,_ = kmeans(dataset, 4)
    idx,_ = vq(dataset, centroids)


if __name__ == "__main__":
    main()