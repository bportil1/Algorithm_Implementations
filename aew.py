import sklearn
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering

import numpy as np

from sklearn import datasets

##### STEP 0 - Preprocess Data

ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data_edited_labels.csv'

ids_train_data = pd.read_csv(ids_train_file)

ids_cut_data = pd.read_csv(ids_train_file)

ids_train_true_labels_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

ids_train_true_labels = pd.read_csv(ids_train_true_labels_file)

ids_train_true_labels = ids_train_true_labels['class'].to_list()

#print(ids_train_true_labels)

ids_test_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Test_data.csv'

ids_test_data = pd.read_csv(ids_test_file)

#print(ids_test_data)

label_encoder = LabelEncoder()

###### train data categorical to numerical

ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

print(ids_cut_data)

ids_cut_data = ids_cut_data.dropna()

print(ids_cut_data)

ids_cut_data['protocol_type'] = label_encoder.fit_transform(ids_cut_data['protocol_type'])

ids_cut_data['service'] = label_encoder.fit_transform(ids_cut_data['service'])

ids_cut_data['flag'] = label_encoder.fit_transform(ids_cut_data['flag'])

ids_cut_labels = ids_cut_data['class']

#ids_train_data['class'] = ids_train_data['class'].fillna('missing')

train_labels = ids_cut_data['class'] #label_encoder.fit_transform(ids_train_data['class'])

#true_labels = label_encoder.fit_transform(ids_train_true_labels)

###### Test data categorical to numerical

ids_test_data['protocol_type'] = label_encoder.fit_transform(ids_test_data['protocol_type'])

ids_test_data['service'] = label_encoder.fit_transform(ids_test_data['service'])

ids_test_data['flag'] = label_encoder.fit_transform(ids_test_data['flag'])

##### STEP 1 - Generate graph data #####

print("Generating Graph Data")

### Graph version of data

ids_train_no_labels = ids_train_data.loc[:, ids_train_data.columns != 'class']

train_data_graph = kneighbors_graph(ids_train_no_labels , n_neighbors=2, mode='distance', metric='minkowski', p=1, include_self=True, n_jobs=-1)

train_cut_data_graph = kneighbors_graph(ids_cut_data.loc[:, ids_cut_data.columns != 'class'], n_neighbors=2, mode='distance', metric='minkowski', p=1, include_self=True, n_jobs=-1)

#test_data_graph = kneighbors_graph(ids_test_data.loc[:, ids_test_data.columns != 'class'], n_neighbors=2, mode='distance', metric='minkowski', p=1, include_self=True, n_jobs=-1)

#print(train_data_graph.shape)

##### STEP 2 - Generate edge weights #####

print("Generating Edge Weights")

### similarity matrix of the graph data
train_graph_similarity_matrix = cosine_similarity(train_data_graph, train_data_graph)

cut_graph_similarity = cosine_similarity(train_cut_data_graph, train_cut_data_graph)

#test_graph_similarity_matrix = cosine_similarity(test_data_graph, test_data_graph)

#print(train_graph_similarity_matrix.shape)

##### STEP 3 - Estimating node labels ##### 

print("Label Propagation")

### Label Prop

label_prop_model = LabelPropagation(n_jobs=-1)

print(cut_graph_similarity)

print(ids_cut_labels)

label_prop_model.fit(cut_graph_similarity, ids_cut_labels)

data_predict = label_prop_model.predict(train_data_graph)

#ids_train_true_labels = ids_train_true_labels.reshape(1,-1)

#print(data_predict.shape)

#print(ids_train_true_labels.shape)

data_predict = np.asarray(data_predict).reshape(1, -1)

true_labels = np.asarray(ids_train_true_labels).reshape(1, -1)

print('#######')

print(data_predict)

print(true_labels)

print("#######")

print(label_prop_model.score(data_predict, true_labels))

#label_prop_prediction = label_prop_model.predict(test_graph_similarity_matrix)

#print(label_prop_prediction.shape)

#print(label_prop_prediction)

### Spectral Clustering

#spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', gamma=.5, n_jobs=-1)

#labels = spectral_clustering.fit_predict(label_prop_model)

#print(labels)
