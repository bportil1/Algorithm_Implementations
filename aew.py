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

##### STEP 0 - Preprocess Data

#ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data_edited_labels.csv'

ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

ids_train_data = pd.read_csv(ids_train_file)

#ids_cut_data = pd.read_csv(ids_train_file)

#ids_train_true_labels_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

#ids_train_true_labels = pd.read_csv(ids_train_true_labels_file)

#ids_train_true_labels = ids_train_true_labels['class'].to_list()

#print(ids_train_true_labels)

#ids_test_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Test_data.csv'

#ids_test_data = pd.read_csv(ids_test_file)

#print(ids_test_data)

label_encoder = LabelEncoder()

###### train data categorical to numerical

ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

train_labels = ids_train_data['class'] #label_encoder.fit_transform(ids_train_data['class'])

##### STEP 1 - Generate graph data #####

print("Generating Graph Data")

### Graph version of data

ids_train_no_labels = ids_train_data.loc[:, ids_train_data.columns != 'class']

train_data_graph = kneighbors_graph(ids_train_no_labels , n_neighbors=5, mode='connectivity', metric='minkowski', p=1, include_self=False, n_jobs=-1)

#print(train_data_graph)

#print(train_data_graph)
col_indices = train_data_graph.indices
row_indices = train_data_graph.indptr
point_weight = train_data_graph.data

similarity_matrix = np.zeros((train_data_graph.shape[0], train_data_graph.shape[0]))

for idx in range(train_data_graph.shape[0]):
              
    point = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])

    gamma = 1
    
    point1 = np.asarray(ids_train_no_labels.loc[[idx]])

    for vertex in train_data_graph.indices[point]:
        point2 = np.asarray(ids_train_no_labels.loc[[vertex]])

        intermed_res = 0

        for feature in range(len(point2[0])):

            intermed_res += (point1[0][feature] - point2[0][feature]) ** 2 / gamma

        similarity_matrix[idx][vertex] = exp(-intermed_res)

##### STEP 2 - Generate edge weights #####

print("Generating Edge Weights")

##### STEP 3 - Estimating node labels ##### 

print("Label Propagation")

### Label Prop

label_prop_model = LabelPropagation(n_jobs=-1)

label_prop_model.fit(similarity_matrix, train_labels)

data_predict = label_prop_model.score(similarity_matrix, train_labels)

print(data_predict)

### Spectral Clustering

spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', gamma=.5, n_jobs=-1)

labels = spectral_clustering.fit_predict(similarity_matrix)
print(labels.labels_)
print(np.count_nonzero(labels.labels_ == train_labels))
print("End of Script")
