# -*- coding: utf-8 -*-
import numpy as np

def load_dataset(file):
    return np.loadtxt(file)

def euclidian(a, b):
    return np.linalg.norm(a-b)

def kmeans(k, epsilon=0, distance="euclidian"):
    history_centroids = []
    if distance == "euclidian":
        dist_method = euclidian
    dataset = load_dataset('durudataset.txt')
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
        print(f"norm: {norm}")
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
