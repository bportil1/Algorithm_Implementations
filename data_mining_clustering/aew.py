import sklearn
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from math import exp
from spread_opt import *
from lower_dim_mappings_utils import *
from sklearn.model_selection import train_test_split
from multiprocessing.sharedctypes import RawArray



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

    return train_data, train_labels, test_data, test_labels

def generate_graph(train_data):
    print("Generating Graph")
    train_data_graph = kneighbors_graph(train_data , n_neighbors=20, mode='connectivity', metric='minkowski', p=1, include_self=False, n_jobs=-1)

    return train_data_graph

def generate_optimal_edge_weights(train_data, train_data_graph, num_iterations): 
    print("Generating Optimal Edge Weights")
    gamma = np.ones(train_data.loc[[0]].shape[1])

    gamma = gradient_descent(.1, num_iterations, .01, train_data, train_data_graph, gamma)

    similarity_matrix = generate_edge_weights(train_data, train_data_graph, gamma)

    return similarity_matrix, gamma

def edge_weight_computation(train_data, train_data_graph , gamma, section):
    
    res = []

    for idx in section:
        point = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])

        point1 = np.asarray(train_data.loc[[idx]])

        for vertex in train_data_graph.indices[point]:

            point2 = np.asarray(train_data.loc[[vertex]])
    
            res.append((idx, vertex, similarity_function(train_data, gamma, idx, vertex)))
            
    return res


def generate_edge_weights(train_data, train_data_graph, gamma):
    print("Generating Edge Weights")

    similarity_matrix = np.zeros((train_data_graph.shape[0], train_data_graph.shape[0]))

    split_data = split(range(train_data_graph.shape[0]), cpu_count())

    with Pool(processes=cpu_count()) as pool:
        edge_weight_res = [pool.apply_async(edge_weight_computation, (train_data, train_data_graph, gamma, section)) for section in split_data]

        edge_weights = [edge_weight.get() for edge_weight in edge_weight_res] 

    for section in edge_weights:
        for weight in section:
            similarity_matrix[weight[0]][weight[1]] = weight[2]

    print("Edge Weight Generation Complete")
    return similarity_matrix

def estimate_node_labels(adjacency_matrix, true_labels):

    label_prop_model = LabelPropagation(n_jobs=-1)

    label_prop_model.fit(adjacency_matrix, true_labels)

    #data_predict = label_prop_model.score(adjacency_matrix, true_labels)

    data_predict = label_prop_model.predict(adjacency_matrix)

    #print("Label Accuracy: ", data_predict)

    return data_predict

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = preprocess_data()

    visualization_tester(train_data, train_labels)

    graph = generate_graph(train_data)

    train_adj_matr, gamma = generate_optimal_edge_weights(train_data, graph, 5)

    #test_graph = generate_graph(test_data)

    #test_adj_matr = generate_edge_weights(test_data, test_graph, gamma)

    predicted_labels = estimate_node_labels(train_adj_matr, train_labels)

    visualization_tester(train_adj_matr, predicted_labels)

