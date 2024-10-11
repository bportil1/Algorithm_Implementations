import sklearn
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from math import exp

import sys

import numpy as np

from sklearn import datasets

np.set_printoptions(threshold=sys.maxsize)


def preprocess_data():
    ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    ids_train_data = pd.read_csv(ids_train_file)

    label_encoder = LabelEncoder()

    ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

    ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

    ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

    train_labels = ids_train_data['class'] 

    ids_train_no_labels = ids_train_data.loc[:, ids_train_data.columns != 'class']


    return ids_train_no_labels, train_labels

def generate_graph(train_data):
    print("Generating Graph")
    train_data_graph = kneighbors_graph(train_data , n_neighbors=20, mode='distance', metric='minkowski', p=1, include_self=False, n_jobs=-1)

    return train_data_graph

def optimize_spread_param(): 

    return 0

def generate_edge_weights(train_data, train_data_graph):
    print("Generating Edge Weights")
    col_indices = train_data_graph.indices
    row_indices = train_data_graph.indptr
    point_weight = train_data_graph.data

    similarity_matrix = np.zeros((train_data_graph.shape[0], train_data_graph.shape[0]))

    for idx in range(train_data_graph.shape[0]):
              
        point = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])

        gamma = 1 

        point1 = np.asarray(train_data.loc[[idx]])

        for vertex in train_data_graph.indices[point]:
            point2 = np.asarray(train_data.loc[[vertex]])

            intermed_res = 0

            for feature in range(len(point2[0])):

                intermed_res += (point1[0][feature] - point2[0][feature]) ** 2 / gamma

            similarity_matrix[idx][vertex] = exp(-intermed_res)

    return similarity_matrix

def estimate_node_labels(adjacency_matrix, true_labels=None):
    label_prop_model = LabelPropagation(n_jobs=-1)

    label_prop_model.fit(adjacency_matrix, true_labels)

    data_predict = label_prop_model.score(adjacency_matrix, true_labels)

    print(data_predict)

    return 0

def measure_accuracy():

    return 0

if __name__ == '__main__':
    data, labels = preprocess_data()
    graph = generate_graph(data)
    adj_matr = generate_edge_weights(data, graph)
    estimate_node_labels(adj_matr, labels)
'''
### Spectral Clustering

spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', gamma=.5, n_jobs=-1)

labels = spectral_clustering.fit_predict(similarity_matrix)
print(labels.labels_)
print(np.count_nonzero(labels.labels_ == train_labels))
print("End of Script")
'''
