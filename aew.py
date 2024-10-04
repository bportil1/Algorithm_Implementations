import sklearn
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation


ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data_edited_labels.csv'

ids_train_data = pd.read_csv(ids_train_file)

ids_test_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Test_data.csv'

ids_test_data = pd.read_csv(ids_test_file)

#print(ids_test_data)

label_encoder = LabelEncoder()

#print(ids_train_data.loc[:, ids_train_data.columns != 'class'])

ids_train_data['class'] = ids_train_data['class'].fillna(-1)

print(ids_train_data)

ids_train_data['protocol_type'] = label_encoder.fit_transform(ids_train_data['protocol_type'])

ids_train_data['service'] = label_encoder.fit_transform(ids_train_data['service'])

ids_train_data['flag'] = label_encoder.fit_transform(ids_train_data['flag'])

train_data_graph = kneighbors_graph(ids_train_data.loc[:, ids_train_data.columns != 'class'], n_neighbors=2, mode='distance', metric='minkowski', p=1, n_jobs=-1)

label_prop_model = LabelPropagation()



print(train_data_graph.toarray())
