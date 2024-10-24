#import sklearn
#import pandas as pd
#from sklearn.neighbors import kneighbors_graph
#from sklearn.preprocessing import LabelEncoder
#from sklearn.semi_supervised import LabelPropagation
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import SpectralClustering
#from math import exp
from spread_opt import *
#from lower_dim_mappings_utils import *
from preprocessing_utils import *
#from sklearn.model_selection import train_test_split
#from multiprocessing.sharedctypes import RawArray
from clustering import *

#import sys

#import numpy as np

#from sklearn import datasets

#from sklearn.decomposition import PCA

#from sklearn.preprocessing import MinMaxScaler

#np.set_printoptions(threshold=sys.maxsize)

def generate_graph(train_data, train_labels):
    print("Generating Graph")

    #train_data = pca_centered(train_data.to_numpy(), .9)
    '''
    n_components = [ idx for idx in range(2, 20, 2)]

    max_connectivity = float('inf')

    max_conn_idx = 0

    min_variance = .85
    
    curr_variance = 0

    for n_cmp in n_components:

        pca = PCA(n_components=n_cmp)

        tmp_train_data = pca.fit_transform(train_data)

        #visualization_tester(train_data, train_labels, 3, 'display')

        curr_variance = np.sum(pca.fit(train_data).explained_variance_ratio_)

        print(pca.fit(train_data).explained_variance_ratio_)

        #train_data = visualization_tester(train_data, train_labels, 3, 'no')
        graph = kneighbors_graph(tmp_train_data, n_neighbors=150, mode='connectivity', metric='euclidean', include_self=False, n_jobs=-1)
        
        G = nx.from_numpy_array(graph)

        components = list(nx.connected_components(G))

        if len(components) < max_connectivity:

            max_connectivity = len(components)

            max_conn_idx = n_cmp

        print(len(components))

        del G

        if curr_variance >= min_variance:
            break

    pca = PCA(n_components=max_conn_idx)
    '''
    pca = PCA(n_components=4)

    train_data = pca.fit_transform(train_data)

    #train_data = visualization_tester(train_data, train_labels, 3, 'no')
    graph = kneighbors_graph(train_data, n_neighbors=150, mode='connectivity', metric='euclidean', include_self=False, n_jobs=-1)

    #if n_cmp > 2:
    #    visualization_tester(graph, train_labels, 3, 'display')


    return graph

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

    #print(similarity_matrix[0][:5])

    similarity_matrix = laplacian_normalization(similarity_matrix)

    #print(similarity_matrix[0][:5])

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
    ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    data_obj = data()

    data_obj.load_data(ids_train_file, 'train')

    data_obj.load_labels('train', from_data=True)

    data_obj.split_data(.2)

    data_obj.encode_categorical('protocol_type', 'data')

    data_obj.encode_categorical('service', 'data')

    data_obj.encode_categorical('flag', 'data')

    data_obj.encode_categorical('class', 'labels')

    data_obj.scale_data('min_max')

    init_path = './results/orig_data_visualization/'

    os.makedirs(init_path, exist_ok=True)

    data_obj.lower_dimensional_embedding(data_obj.train_data, 'train', 'Original Train Data: 3-Dimensions', init_path, 'data')

    data_obj.lower_dimensional_embedding(data_obj.test_data, 'test', 'Original Test Data: 3-Dimensions', init_path, 'data')

    data_obj.generate_graphs('train')

    data_obj.generate_graphs('test')

    init_path = './results/orig_data_visualization/graphs/'

    os.makedirs(init_path, exist_ok=True)

    data_obj.lower_dimensional_embedding(data_obj.train_graph ,'train', 'Original Train Graph: 3-Dimensions', init_path, 'graph')

    data_obj.lower_dimensional_embedding(data_obj.test_graph, 'test', 'Original Test Graph: 3-Dimensions', init_path, 'graph')

    aew_train = aew(data_obj.train_graph, data_obj.train_data)

    aew_train.generate_optimal_edge_weights(1)

    aew_test = aew(data_obj.test_graph, data_obj.test_data, aew_train.gamma)

    aew_test.generate_edge_weights()

    num_components = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 35, 40]

    for num_comp in num_components:

        print("Current number of components: ", num_com)

        data_obj.train_projection, _ = data_obj.downsize_data(data_obj.train_graph, 'train', num_comp)

        data_obj.test_projection, _ = data_obj.downsize_data(data_obj.test_graph, 'test', num_comp)
        init_path = './results/orig_data_visualization/num_comp_' + str(num_comp) + '/'

        os.makedirs(init_path, exist_ok=True)

        data_obj.lower_dimensional_embedding(data_obj.train_projection, 'train', 'Train Mappings Base: 3-Dimensions', init_path, proj=True)

        data_obj.lower_dimensional_embedding(data_obj.test_projection, 'test', 'Test Mappings Base: 3-Dimensions', init_path, proj=True)

        clustering = clustering(data_obj.train_projection, data_obj.train_labels,
                                data_obj.test_data, data_obj.test_labels,
                                workers = -1)

        clustering.generate_clustering()

    '''
    train_data, train_labels, test_data, test_labels = preprocess_ids_data
    train_labels =  train_labels['class']

    test_labels = test_labels['class']
   
    ######Usually display here

    train_data = pd.DataFrame(MinMaxScaler().fit_transform(train_data))

    test_data = pd.DataFrame(MinMaxScaler().fit_transform(test_data))

    visualization_tester(train_data, train_labels, 3, 'no')

    visualization_tester(test_data, test_labels, 3, 'no')

    estimate_node_labels(train_data, train_labels, test_data, test_labels)

    graph = generate_graph(train_data, train_labels)

    train_adj_matr, gamma = generate_optimal_edge_weights(train_data, graph, 10)

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

    num_components = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 35, 40]

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
            
            print("Spectral Clustering")

            hyper_para_list = [2, 3, 4, 5] ,# 6, 7, 8, 9, 10, 15, 20]

            for hyper_para in hyper_para_list:

                print("Number of Clusters: ", hyper_para)

                clustering_model = SpectralClustering( n_clusters = hyper_para, assign_labels = 'discretize')

                train_pred = clustering_model.fit_predict(train_data)

                print("Spectral Train Accuracy: ", accuracy_score(train_pred, train_labels))

                test_pred = clustering_model.fit_predict(test_data)

                print("Spectral Test Accuracy: ", accuracy_score(test_pred, test_labels))
            
            print("Knn Clustering")

            hyper_para_list = [2, 3, 4, 5] ,# 6, 7, 8, 9, 10, 15, 20, 50, 100]

            for hyper_para in hyper_para_list:

                print("Number of Clusters: ", hyper_para)

                clustering_model = KMeans(n_clusters=hyper_para).fit(train_data)

                train_pred = clustering_model.predict(train_data)

                print("K-means Train Accuracy: ", accuracy_score(train_pred, train_labels))

                clustering_model = KMeans(n_clusters=hyper_para).fit(test_data)

                test_pred = clustering_model.predict(test_data)

                print("K-means Test Accuracy: ", accuracy_score(test_pred, test_labels))
            
            print("Agglomerative clustering")
            
            hyper_para_list = [2, 3, 4, 5] #, 6, 7, 8, 9, 10, 15, 20]

            for hyper_para in hyper_para_list:

                print("Number of Clusters: ", hyper_para)

                clustering_model = AgglomerativeClustering(n_clusters= hyper_para, linkage= 'ward')

                train_pred = clustering_model.fit_predict(train_data)

                print("Agglo Train Accuracy: ", accuracy_score(train_pred, train_labels))

                test_pred = clustering_model.fit_predict(test_data)

                print("Agglo Test Accuracy: ", accuracy_score(test_pred, test_labels))
        '''
