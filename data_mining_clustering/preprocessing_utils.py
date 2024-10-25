from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    TSNE,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from time import time
from sklearn.neighbors import kneighbors_graph
import plotly.express as px

class data():
    def __init__(self, train_data = None, train_labels = None, test_data = None, test_labels = None, output_path = None):
        self.train_data = train_data
        self.train_graph = None
        self.train_labels = train_labels
        self.train_projection = None
        self.test_data = pd.DataFrame()
        self.test_graph = None
        self.test_labels = test_labels
        self.test_projection = None

    def scale_data(self, scaling):
        if scaling == 'standard':
            self.train_data[[col for col in self.train_data]] = StandardScaler().fit_transform(self.train_data[[col for col in self.train_data]])
            if not self.test_data.empty:
                self.test_data[[col for col in self.test_data]] = StandardScaler().fit_transform(self.test_data[[col for col in self.test_data]])
        elif scaling == 'min_max':
            min_max_scaler = MinMaxScaler()
            self.train_data[[col for col in self.train_data]] = min_max_scaler.fit_transform(self.train_data[[col for col in self.train_data]])
            if not self.test_data.empty:
                self.test_data[[col for col in self.test_data]] = min_max_scaler.fit_transform(self.test_data[[col for col in self.test_data]])

        else:
            print("Scaling arg not supported")
        
    def encode_categorical(self, column_name, target_set):
        label_encoder = LabelEncoder()

        if target_set == 'data': 
            label_encoder.fit(list(self.train_data[column_name])+list(self.test_data[column_name])) 
            self.train_data[column_name] = label_encoder.transform(self.train_data[column_name])
            self.test_data[column_name] = label_encoder.transform(self.test_data[column_name])
        elif target_set == 'labels':
            label_encoder.fit(list(self.train_labels[column_name])+list(self.test_labels[column_name])) 
            self.train_labels[column_name] = label_encoder.transform(self.train_labels[column_name])
            self.test_labels[column_name] = label_encoder.transform(self.test_labels[column_name])

    def load_data(self, datapath, data_type):
        if data_type == 'train':
            self.train_data = pd.read_csv(datapath)
            self.train_data = self.train_data.head(5000)
        elif data_type == 'test':
            self.test_data = pd.read_csv(datapath)

    def load_labels(self, data_type, datapath=None, from_data = False):
    
        if from_data:
            if data_type == 'train':
                self.train_labels = pd.DataFrame(self.train_data['class'], columns=['class'])
                #self.train_labels.columns = ['class']
                self.train_data = self.train_data.loc[:, self.train_data.columns != 'class']
            elif data_type == 'test':
                self.test_labels['class'] = pd.DataFrame(self.test_data['class'], columns=['class'])
                #self.test_labels.columns = ['class']
                self.test_data = self.test_data.loc[:, self.test_data.columns != 'class']
        else:
            if data_type == 'train':
                self.train_labels = pd.read_csv(datapath)
            elif data_type == 'test':
                self.test_labels = pd.read_csv(datapath)

    def split_data(self, split_size):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.train_data, self.train_labels, test_size = split_size)

        self.reset_indices()

    def reset_indices(self):
        self.train_data = self.train_data.reset_index(drop=True)
        self.train_labels = self.train_labels.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)
        self.test_labels = self.test_labels.reset_index(drop=True)


    def get_embeddings(self, num_components, embedding_subset = None):
        embeddings = {
            "Truncated SVD embedding": TruncatedSVD(n_components=num_components),
            #"Standard LLE embedding": LocallyLinearEmbedding(
            #    n_neighbors=n_neighbors, n_components=num_components, method="standard", 
            #    eigen_solver='dense', n_jobs=-1
            #),
            "Random Trees embedding": make_pipeline(
                RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0, n_jobs=-1),
                TruncatedSVD(n_components=num_components),
            ),
            #"t-SNE embedding": TSNE(
            #        n_components=num_components,
            #    max_iter=500,
            #    n_iter_without_progress=150,
            #    n_jobs=-1,
            #    random_state=0,
            #),
            "PCA": PCA(n_components=3),
        }
        if embedding_subset == None:
            return embeddings
        else:
            out_dict = {}
            for key, value in enumerate(embeddings):
                out_dict[key] = value
            return out_dict

    def downsize_data(self, data, data_type, num_components):
        if data_type == 'train':
            labels = self.train_labels
        elif data_type == 'test':
            labels = self.test_labels

        embeddings = self.get_embeddings(num_components)

        projections, timing = {}, {}
        for name, transformer in embeddings.items():
            print(f"Computing {name}...")
            start_time = time()
            projections[name] = transformer.fit_transform(data, labels)
            timing[name] = time() - start_time

        return projections, timing 

    def generate_graphs(self, data_type):
        if data_type == 'train':
            self.train_graph = kneighbors_graph(self.train_data, n_neighbors=150, mode='connectivity', metric='euclidean', include_self=False, n_jobs=-1)
        elif data_type == 'test':
            self.test_graph = kneighbors_graph(self.test_data, n_neighbors=150, mode='connectivity', metric='euclidean', include_self=False, n_jobs=-1)

    def lower_dimensional_embedding(self, data, data_type, passed_title, path):
        if data_type == 'train':
            labels = self.train_labels
        elif data_type == 'test': 
            labels = self.test_labels

        embeddings = self.get_embeddings(3)
        projections, timing = self.downsize_data(data, 'train', 3) 
        
        for name in timing:
            title = f"{name} (time {timing[name]:.3f}s  {passed_title})"
            file_path = str(path) + str(name) + '.html'
            self.plot_embedding(projections[name], labels, title, file_path)

    def plot_embedding(self, data, labels, title, path):
        cdict = { 0: 'blue', 1: 'red'}

        df = pd.DataFrame({ 'x1': data[:,0],
                            'x2': data[:,1],
                            'x3': data[:,2],
                            'label': labels['class'] })
    
        for label in np.unique(labels):
            idx = np.where(labels == label)
            fig = px.scatter_3d(df, x='x1', y='x2', z='x3',
                                color='label', color_discrete_map=cdict,
                                opacity=.4)

            fig.update_layout(
                title = title
            )

        fig.write_html(path, div_id = title)
        #fig.show()

def preprocess_ids_data():
    #ds_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

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

