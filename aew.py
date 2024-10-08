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

ids_test_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Test_data.csv'

ids_test_data = pd.read_csv(ids_test_file)

#print(ids_test_data)

label_encoder = LabelEncoder()

###### train data categorical to numerical

ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

ids_train_data['class'] = ids_train_data['class'].fillna('missing')

train_labels = label_encoder.fit_transform(ids_train_data['class'])

###### Test data categorical to numerical

ids_test_data['protocol_type'] = label_encoder.fit_transform(ids_test_data['protocol_type'])

ids_test_data['service'] = label_encoder.fit_transform(ids_test_data['service'])

ids_test_data['flag'] = label_encoder.fit_transform(ids_test_data['flag'])

#ids_test_data['class'] = ids_test_data['class'].fillna('missing')

#test_labels = label_encoder.fit_transform(ids_test_data['class'])

##### STEP 1 - Generate graph data #####

### Graph version of data
train_data_graph = kneighbors_graph(ids_train_data.loc[:, ids_train_data.columns != 'class'], n_neighbors=2, mode='distance', metric='minkowski', p=1, include_self=True, n_jobs=-1)

test_data_graph = kneighbors_graph(ids_test_data, n_neighbors=2, mode='distance', metric='minkowski', p=1, include_self=True, n_jobs=-1)

##### STEP 2 - Generate edge weights #####

### similarity matrix of the graph data
train_graph_similarity_matrix = cosine_similarity(train_data_graph, train_data_graph)

test_graph_similarity_matrix = cosine_similarity(test_data_graph, test_data_graph)

##### STEP 3 - Estimating node labels ##### 

### Label Prop

label_prop_model = LabelPropagation()

label_prop_model.fit(train_graph_similarity_matrix, train_labels)

label_prop_prediction = label_prop_model.predict(test_graph_similarity_matrix)

print(label_prop_prediction)

### Spectral Clustering

spectral_clustering = SpectralClustering(n_clusters=3, affinity='rbf', gamma=.5)

labels = spectral_clustering.fit_predict()
