import sklearn
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from math import exp
from spread_opt import *
from sklearn.model_selection import train_test_split

import sys

import numpy as np

from sklearn import datasets

np.set_printoptions(threshold=sys.maxsize)

def preprocess_data():
    #ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Test_data.csv'

    ids_train_data = pd.read_csv(ids_train_file)

    label_encoder = LabelEncoder()

    ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

    ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

    ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

    ids_train_data = ids_train_data.sample(frac=1)

    train_set, test_set = train_test_split(ids_train_data, test_size = .2)

    train_set = train_set.reset_index(drop=True)

    test_set = test_set.reset_index(drop=True)

    train_data = train_set.loc[:, train_set.columns != 'class']

    train_labels = train_set['class']

    test_data = test_set.loc[:, test_set.columns != 'class']

    test_labels = test_set['class']

    #print(train_data)
    #print(train_labels)
    #print(test_data)
    #print(test_labels)

    return train_data, train_labels, test_data, test_labels

def generate_graph(train_data):
    print("Generating Graph")
    train_data_graph = kneighbors_graph(train_data , n_neighbors=20, mode='connectivity', metric='minkowski', p=1, include_self=False, n_jobs=-1)

    return train_data_graph

def generate_optimal_edge_weights(train_data, train_data_graph): 
    print("Generating Optimal Edge Weights")
    gamma = np.ones(train_data.loc[[0]].shape[1])

    #similarity_matrix = generate_edge_weights(train_data, train_data_graph, gamma)

    gamma = gradient_descent(.1, 10, .01, train_data, train_data_graph, gamma)

    #gamma = gradient_descent(1, 1, 1, train_data, train_data_graph, gamma)

    similarity_matrix = generate_edge_weights(train_data, train_data_graph, gamma)

    return similarity_matrix, gamma

def generate_edge_weights(train_data, train_data_graph, gamma):
    print("Generating Edge Weights")
    col_indices = train_data_graph.indices
    row_indices = train_data_graph.indptr
    point_weight = train_data_graph.data

    similarity_matrix = np.zeros((train_data_graph.shape[0], train_data_graph.shape[0]))

    for idx in range(train_data_graph.shape[0]):

        progress = idx/train_data_graph.shape[0]

        if int(progress*100) % 5 == 0:

            print("Progress: ", str(progress*100), "%")

        point = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])

        point1 = np.asarray(train_data.loc[[idx]])

        for vertex in train_data_graph.indices[point]:

            point2 = np.asarray(train_data.loc[[vertex]])

            similarity_matrix[idx][vertex] = similarity_function(train_data, gamma, idx, vertex)
    print("Edge Weight Generation Complete")
    return similarity_matrix

def estimate_node_labels(adjacency_matrix, true_labels, test_data, test_labels):

    #print(adjacency_matrix.shape())
    #print(true_labels)
    #print(test_data.shape())
    #print(test_labels)

    label_prop_model = LabelPropagation(n_jobs=-1)

    #label_prop_model.fit(adjacency_matrix, true_labels)

    label_prop_model.fit(test_data, test_labels)

    data_predict = label_prop_model.score(test_data, test_labels)

    print("Label Accuracy: ", data_predict)

    return 0

def measure_accuracy():

    return 0

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = preprocess_data()
    graph = generate_graph(train_data)
    train_adj_matr, gamma = generate_optimal_edge_weights(train_data, graph)

    test_graph = generate_graph(test_data)

    test_adj_matr = generate_edge_weights(test_data, test_graph, gamma)

    estimate_node_labels(train_adj_matr, train_labels, test_adj_matr, test_labels)
'''
### Spectral Clustering

spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', gamma=.5, n_jobs=-1)

labels = spectral_clustering.fit_predict(similarity_matrix)
print(labels.labels_)
print(np.count_nonzero(labels.labels_ == train_labels))
print("End of Script")
'''
