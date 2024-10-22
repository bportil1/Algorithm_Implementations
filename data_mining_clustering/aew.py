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

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=sys.maxsize)

def generate_graph(train_data, train_labels):
    print("Generating Graph")

    #train_data = pca_centered(train_data.to_numpy(), .9)
    
    pca = PCA(n_components=2)

    train_data = pca.fit_transform(train_data)
    
    train_data = MinMaxScaler().fit_transform(train_data)

    #visualization_tester(train_data, train_labels, 3, 'display')

    print(pca.fit(train_data).explained_variance_ratio_)

    #print(len(train_data))
    #print(len(train_data[0]))

    #train_data = visualization_tester(train_data, train_labels, 3, 'no')
    train_data_graph = kneighbors_graph(train_data, n_neighbors=150, mode='connectivity', metric='euclidean', include_self=False, n_jobs=-1)

    visualization_tester(train_data_graph, train_labels, 3, 'display')


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

    print(similarity_matrix[0][:5])

    similarity_matrix = laplacian_normalization(similarity_matrix)

    print(similarity_matrix[0][:5])

    print("Edge Weight Generation Complete")
    
    return similarity_matrix

def estimate_node_labels(train_data, train_labels, test_data, test_labels):

    label_prop_model = LabelPropagation(n_jobs=-1)

    label_prop_model.fit(train_data, train_labels)

    data_predict = label_prop_model.score(test_data, test_labels)

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

    #train_labels = flat
    '''
    #clustering2 = clustering(train_vectors.loc[:, train_vectors.columns != 'index'], train_labels, test_vectors.loc[:, test_vectors.columns != 'index'], test_labels, data_path)

    #clustering2.clustering_training()
    
    train_labels =  train_labels['class']

    test_labels = test_labels['class']
   
    ######Usually display here

    visualization_tester(train_data, train_labels, 3, 'display')

    visualization_tester(test_data, test_labels, 3, 'no')

    estimate_node_labels(train_data, train_labels, test_data, test_labels)

    graph = generate_graph(train_data, train_labels)

    train_adj_matr, gamma = generate_optimal_edge_weights(train_data, graph, 5)

    graph = rewrite_edges(graph, train_adj_matr)

    test_graph = generate_graph(test_data, test_labels)

    test_adj_matr = generate_edge_weights(test_data, test_graph, gamma)

    test_graph = rewrite_edges(test_graph, test_adj_matr)

    #estimate_node_labels(graph, train_labels, test_graph, test_labels)

    graph = graph.toarray()

    #print(len(graph))

    #print(len(graph[0]))

    test_graph = test_graph.toarray()

    #print(test_graph[:5])

    num_components = [3, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 75, 100]

    #num_components = [18, 20, 25, 30, 35, 40, 75, 100]

    graph = np.asarray(graph)

    for num_com in num_components:

        print("Current number of components: ", num_com)

        projections = visualization_tester(graph, train_labels, num_com, 'no')

        test_proj = visualization_tester(test_graph, test_labels, num_com, 'no')

        for projection in projections.keys():
   
            print("Projection Type: ", projection)

            train_data = projections[projection]

            test_data = test_proj[projection]

            estimate_node_labels(train_data, train_labels, test_data, test_labels)

            ### usually displaay here

            visualization_tester(train_data, train_labels, 3, 'no')

            #twod_data = visualization_tester(train_data, train_labels, 2, 'no')

            #visualization_tester(test_data, test_labels, 3, 'display')
        
            #G = nx.from_numpy_array(graph)
    
            #components = list(nx.connected_components(G))

            #print(len(components))
            '''
            for component in components:

                subgraph = G.subgraph(component)
                subgraph_adj = nx.to_numpy_array(subgraph)

                hyper_para_list = np.arange(2, 31, step = 1)

                if len(subgraph_adj) < 2:
                    continue


                for hyper_para in hyper_para_list:

                    clustering_model = SpectralClustering( n_clusters = hyper_para, assign_labels = 'discretize').fit(subgraph_adj)

                    train_pred = clustering_model.fit_predict(subgraph_adj)

                    cluster_Labels = clustering_model.labels_

                    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
                    # core_samples_mask[clustering.core_sample_indices_] = True

                    # Number of clusters in labels, ignoring noise if present.
                    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
                    n_noise_ = list(cluster_Labels).count(-1)
                    # Plot result

                    # Black removed and is used for noise instead.
                    unique_labels = set(cluster_Labels)
                    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # Black used for noise.
                            col = [0, 0, 0, 1]

                        class_member_mask = cluster_Labels == k

                        xy = twod_data[projection][class_member_mask]
                        plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=5,
                        )
                        #fig = plt.gcf()
                        plt.show()

                


                    #print(len(train_pred))

                    #quart = (len(train_pred//4))
                    #print(train_pred[:quart])

                    #print("Accuracy: ", accuracy_score(train_pred, train_labels))

                    #test_pred = clustering_model.fit_predict(test_data)

                    #print("Accuracy: ", accuracy_score(test_pred, test_labels))

            '''
            
            print("Knn Clustering")

            hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100]

            for hyper_para in hyper_para_list:

                clustering_model = KMeans(n_clusters=hyper_para).fit(graph)

                train_pred = clustering_model.predict(graph)

                print("K-means Train Accuracy: ", accuracy_score(train_pred, train_labels))

                clustering_model = KMeans(n_clusters=hyper_para).fit(test_graph)

                test_pred = clustering_model.predict(test_data)

                print("K-means Test Accuracy: ", accuracy_score(test_pred, test_labels))
            
            print("Agglomerative clustering")
            
            hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

            for hyper_para in hyper_para_list:

                clustering_model = AgglomerativeClustering(n_clusters= hyper_para, linkage= 'ward')

                train_pred = clustering_model.fit_predict(train_data)

                print("Train Accuracy: ", accuracy_score(train_pred, train_labels))

                test_pred = clustering_model.fit_predict(test_data)

                print("Accuracy: ", accuracy_score(test_pred, test_labels))
            
            print("DBSCAN Clustering")

            hyper_para_list = np.arange(5,150 , step = 5)

            m_samp = np.arange(5,150 , step = 5)

            for hyper_para in hyper_para_list:

                #for m in m_samp:
                clustering_model = DBSCAN(eps=hyper_para/100, min_samples=50, n_jobs = -1)

                train_pred = clustering_model.fit_predict(train_data)   

                print("DBSCAN Train Accuracy: ", accuracy_score(train_pred, train_labels))

                test_pred = clustering_model.fit_predict(test_data)

                print("DBSCAN Test Accuracy: ", accuracy_score(test_pred, test_labels))
            
    
