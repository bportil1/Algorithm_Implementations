import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from functools import reduce

from multiprocessing import cpu_count

import networkx as nx

from node2vec import Node2Vec

class data():
    def __init__(self, train_data = None, train_labels = None, test_data = None, test_labels = None, output_path = None):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.output_path = output_path

    def scale_data(self, scaling):
        if scaling == 'standard':
            self.train_data[[col for col in self.train_data]] = StandardScaler().fit_transform(self.train_data[[col for col in self.train_data]])
            if self.test_data != None:
                self.test_data[[col for col in self.test_data]] = StandardScaler().fit_transform(self.test_data[[col for col in self.test_data]])
        elif scaling == 'min_max':
            min_max_scaler = MinMaxScaler()
            self.train_data[[col for col in self.train_data]] = min_max_scaler.fit_transform(self.train_data[[col for col in self.train_data]])
            if self.test_data != None:
                self.test_data[[col for col in self.test_data]] = min_max_scaler.fit_transform(self.test_data[[col for col in self.test_data]])

        else:
            print("Scaling arg not supported")
        
    def encode_categorical(self, set_type, column_name):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.train_data)

        self.train_data[data_type] = label_encoder.transform(self.train_data[data_type])
        self.test_data[data_type] = label_encoder.transform(self.test_data[data_type])

    def load_data(self, datapath, data_type):
        if data_type == 'train':
            self.train_data = pd.read_csv(datapath)
        elif data_type == 'test':
            self.test_data = pd.read_csv(datapath)

    def load_labels(self, data_type, datapath, from_data = False):
    
        if from_data:
            if data_type == 'train':
                self.train_labels = self.train_labels['class']
                self.train_data = self.train_data.loc[:, self.train_data.columns != 'class']
            elif data_type == 'test':
                self.test_labels = self.test_labels['class']
                self.test_data = self.test_data.loc[:, self.test_data.columns != 'class']
        else:
            if data_type == 'train':
                self.train_labels = pd.read_csv(datapath)
            elif data_type == 'test':
                self.test_labels = pd.read_csv(datapath)

    def split_data(self, split_size):
        self.train_data, self.train_labels, self.test_data, self.test_labels = train_test_split(self.train_data, self.train_labels, test_size = split_size)



def preprocess_ids_data():
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

    train_labels = train_set[['class']].copy()

    train_labels['class'].replace(['normal', 'anomaly'], [0,1], inplace=True)

    train_labels.reset_index(inplace=True)

    #train_labels = train_labels.values.tolist()
    
    #train_labels = flatten_list(train_labels)

    test_data = test_set.loc[:, test_set.columns != 'class']

    test_labels = test_set[['class']].copy()

    test_labels['class'].replace(['normal', 'anomaly'], [0,1], inplace=True)

    test_labels.reset_index(inplace=True)

    #test_labels = test_labels.values.tolist()

    #test_labels = flatten_list(test_labels)

    #return train_data.head(500), train_labels.head(500), test_data.head(200), test_labels.head(200)

    return train_data, train_labels, test_data, test_labels

