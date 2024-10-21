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
from preprocessing_utils import *
from sklearn.model_selection import train_test_split
from multiprocessing.sharedctypes import RawArray
from clustering import *

import sys

import numpy as np

from sklearn import datasets

np.set_printoptions(threshold=sys.maxsize)

def generate_graph(train_data):
    print("Generating Graph")
    train_data_graph = kneighbors_graph(train_data , n_neighbors=100, mode='distance', metric='euclidean', p=2, include_self=False, n_jobs=-1)

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

def min_max_scaling(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return ((matrix - min_val) / (max_val - min_val)) + 1

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


    #norm  = np.linalg.norm(similarity_matrix)

    print("Edge Weight Generation Complete")
    #return similarity_matrix/norm
    #return min_max_scaling(similarity_matrix)
    return similarity_matrix

def estimate_node_labels(train_data, train_labels, test_data, test_labels):

    label_prop_model = LabelPropagation(n_jobs=-1)

    label_prop_model.fit(train_data, train_labels)

    data_predict = label_prop_model.score(test_data, test_labels)

    #data_predict = label_prop_model.predict(adjacency_matrix)

    print("LabelProp Accuracy: ", data_predict)

    return data_predict

def rewrite_edges(graph, weights):
    
    rows, cols = graph.nonzero()

    for idx in range(len(rows)):
        row = rows[idx]
        col = cols[idx]
        graph[row, col] = weights[row, col]

    return graph


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = preprocess_ids_data()

    #data_path = './orig_data/'

    print('#####Initial Data#####')

    #clustering1 = clustering(train_data, train_labels, test_data, test_labels, data_path)

    #clustering1.clustering_training()

    #print(train_data.iloc[0])

    #print(train_labels.iloc[0])
    '''
    graph = generate_graph(train_data)
    #print(graph)

    train_adj_matr, gamma = generate_optimal_edge_weights(train_data, graph, 1)
    
    graph = rewrite_edges(graph, train_adj_matr)
    #print(graph)

    #train_vectors = inferN2V(graph)

    test_graph = generate_graph(test_data)

    test_adj_matr = generate_edge_weights(test_data, test_graph, gamma)

    test_graph = rewrite_edges(test_graph, test_adj_matr)
    '''
    #test_vectors = inferN2V(test_graph)
    
    #train_vectors, test_vectors = inferN2V(graph, test_graph)

    #print(train_labels)
    
    '''
    train_vectors = loadN2V('train')

    train_labels['index'] = train_labels['index'].astype(int)

    train_vectors['index'] = train_vectors['index'].astype(int)

    train_vectors = train_vectors.merge(train_labels[['index', 'class']], how='left', on='index')

    train_labels = train_vectors['class']

    train_labels = train_labels.values.tolist()

    #train_labels = flatten_list(train_labels)

    train_vectors = train_vectors.loc[:, train_vectors.columns != 'class']

    test_vectors = loadN2V('test')

    test_labels['index'] = test_labels['index'].astype(int)
    
    test_vectors['index'] = test_vectors['index'].astype(int)

    test_vectors = test_vectors.merge(test_labels[['index', 'class']], how='left', on='index')

    test_labels = test_vectors['class']

    test_labels = test_labels.values.tolist()

    #test_labels = flatten_list(test_labels)

    #print(test_vectors)

    test_vectors = test_vectors.loc[:, test_vectors.columns != 'class']

    #print(train_vectors)

    #print(train_labels)

    #print(test_vectors)

    #print(test_labels)

    train_vectors.columns = train_vectors.columns.astype(str)

    test_vectors.columns = test_vectors.columns.astype(str)

    #predicted_labels = estimate_node_labels(train_adj_matr, train_labels)

    print('#####AEW Data#####')

    estimate_node_labels(train_vectors.loc[:, train_vectors.columns != 'index'], train_labels, test_vectors.loc[:, test_vectors.columns != 'index'], test_labels) 

    #visualization_tester(train_adj_matr, predicted_labels)
    data_path = './after_aew/'
    '''
    #clustering2 = clustering(train_vectors.loc[:, train_vectors.columns != 'index'], train_labels, test_vectors.loc[:, test_vectors.columns != 'index'], test_labels, data_path)

    #clustering2.clustering_training()
    
    train_labels =  train_labels['class']

    test_labels = test_labels['class']
   
    train_3d = visualization_tester(train_data, train_labels)

    test_3d = visualization_tester(test_data, test_labels)

    estimate_node_labels(train_data, train_labels, test_data, test_labels)

    #projections = visualization_tester(graph, train_labels)

    #test_proj = visualization_tester(test_graph, test_labels)

    #print(projections.keys())

    #for projection in projections.keys():
    for projection in train_3d.keys():

        print(projection)

        train_data = train_3d[projection]

        test_data = test_3d[projection]

        graph = generate_graph(train_data)

        print(test_data)


        train_data = pd.DataFrame(data=train_data[0:,0:],
                                  columns=['0', '1', '2'])

        test_data = pd.DataFrame(data=test_data[0:,0:],
                                  columns=['0', '1', '2'])

        print(test_data)


        train_adj_matr, gamma = generate_optimal_edge_weights(train_data, graph, 1)

        graph = rewrite_edges(graph, train_adj_matr)

        test_graph = generate_graph(test_data)

        test_adj_matr = generate_edge_weights(test_data, test_graph, gamma)

        test_graph = rewrite_edges(test_graph, test_adj_matr)

        graph = graph.toarray()

        test_graph = test_graph.toarray()

        '''
        hyper_para_list = np.arange(2, 31, step = 1)

        for hyper_para in hyper_para_list:

            clustering_model = SpectralClustering( n_clusters = hyper_para, affinity='precomputed_nearest_neighbors', assign_labels = 'discretize')

            train_pred = clustering_model.fit_predict(graph)

            print("Accuracy: ", accuracy_score(train_pred, train_labels))

            test_pred = clustering_model.fit_predict(test_graph)

            print("Accuracy: ", accuracy_score(test_pred, test_labels))
        '''

        estimate_node_labels(train_data, train_labels, test_data, test_labels)

        train_data = graph

        test_data = test_graph

        estimate_node_labels(train_data, train_labels, test_data, test_labels)

        hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100]

        for hyper_para in hyper_para_list:

            clustering_model = KMeans(n_clusters=hyper_para).fit(train_data)

            train_pred = clustering_model.predict(train_data)

            print("K-means Train Accuracy: ", accuracy_score(train_pred, train_labels))

            test_pred = clustering_model.predict(test_data)

            print("K-means Test Accuracy: ", accuracy_score(test_pred, test_labels))
        
        '''
        hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        for hyper_para in hyper_para_list:

            clustering_model = AgglomerativeClustering(n_clusters= hyper_para, linkage= 'ward').fit(train_data)

            train_pred = clustering_model.predict(train_data)

            print("Accuracy: ", accuracy_score(train_pred, train_labels))

            #test_pred = clustering_model.predict(test_graph)

            #print("Accuracy: ", accuracy_score(test_pred, test_labels))
        '''

        hyper_para_list = np.arange(5,150 , step = 5)

        for hyper_para in hyper_para_list:

            clustering_model = DBSCAN(eps=hyper_para/100, min_samples=2)

            train_pred = clustering_model.fit_predict(train_data)   

            print("DBSCAN Train Accuracy: ", accuracy_score(train_pred, train_labels))

            test_pred = clustering_model.fit_predict(test_data)

            print("DBSCAN Test Accuracy: ", accuracy_score(test_pred, test_labels))



