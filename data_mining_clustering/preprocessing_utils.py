import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from functools import reduce

from multiprocessing import cpu_count

import networkx as nx

from node2vec import Node2Vec

def scale_data(self, scaling, train_data, train_labels):
    if scaling == 'standard':
        self.train_data[[col for col in self.train_data]] = StandardScaler().fit_transform(self.train_data[[col for col in self.train_data]], self.test_data)
    elif scaling == 'min_max':
        min_max_scaler = MinMaxScaler()
        self.train_data[[col for col in self.train_data]] = min_max_scaler.fit_transform(self.train_data[[col for col in self.train_data]], self.test_data)
    else:
        print("Entered scaling arg not supported")
    return

def flatten_list(nested_list):
    return  reduce(lambda x,y: x+y, nested_list)

def preprocess_ids_data():
    ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

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

    return train_data, train_labels, test_data, test_labels

def trainG2V(train_graphs, train_labels, train_names, param):

    ## Create WL hash word documents
    print('Creating WL hash words for training set')
    train_documents = createWLhash(train_graphs, param)

    ## Shuffling of the data
    ##print('Shuffling data')
    ##train_corpus, train_labels, train_names = DocShuffle(train_documents, train_labels, train_names)

    ## Doc2Vec model training
    ##print('D2V training')
    d2v_model = trainD2Vmodel(train_corpus, param)

    return d2v_model

def inferG2V(test_graphs, test_labels, test_names, param):
    ## Create WL hash word documents for testing set
    print('Creating WL hash words for testing set')
    test_documents = createWLhash(test_graphs, param)

    ## Doc2Vec inference
    print('Doc2Vec inference')
    test_vector = np.array([d2v_model.infer_vector(d.words) for d in test_corpus])

    return test_vector, test_labels, test_names

def inferN2V(train_graph, test_graph):

    #print(graph)
    
    train_graph = nx.from_scipy_sparse_array(train_graph)

    #print(graph2.edges(data=True))

    print("establishing n2v model")

    #print(graph2)

    node2vec = Node2Vec(train_graph, dimensions=40, walk_length=75, num_walks=120, workers=4)

    print("Fitting")

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("saving")

    model.wv.save_word2vec_format("aew_train_40d_embeddings_node2vec.txt")
    
    vectors = []
    line1 = True 
    with open("aew_train_40d_embeddings_node2vec.txt", 'r')  as file:
        for line in file:
            if line1:
                line1 = False
                continue
            
            row = line.strip().split(" ")
            vectors.append(row)

    cols = ['index']

    train_vector_data = [idx for idx in range(40)]

    cols.extend(train_vector_data)

    train_vector_data = pd.DataFrame(vectors, columns=cols)

    test_graph = nx.from_scipy_sparse_array(test_graph)

    #print(graph2.edges(data=True))

    print("establishing n2v model")

    #print(graph2)

    node2vec = Node2Vec(test_graph, dimensions=40, walk_length=75, num_walks=120, workers=4)

    print("Fitting")

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("saving")

    model.wv.save_word2vec_format("aew_test_40d_embeddings_node2vec.txt")

    vectors = []
    line1 = True
    with open("aew_test_40d_embeddings_node2vec.txt", 'r')  as file:
        for line in file:
            if line1:
                line1 = False
                continue

            row = line.strip().split(" ")
            vectors.append(row)

    cols = ['index']

    test_vector_data = [idx for idx in range(40)]

    cols.extend(test_vector_data)

    test_vector_data = pd.DataFrame(vectors, columns=cols)

    return train_vector_data, test_vector_data

def loadN2V(ds_type):
    if ds_type == 'train':
        file = "aew_train_3d_embeddings_node2vec.txt"
    else:
        file = "aew_test_3d_embeddings_node2vec.txt"

    vectors = []
    line1 = True 
    with open(file, 'r')  as file1:
        for line in file1:
            if line1:
                line1 = False
                continue
            
            row = line.strip().split(" ")
            vectors.append(row)

    cols = ['index']

    vector_data = [idx for idx in range(40)]

    cols.extend(vector_data)

    return pd.DataFrame(vectors, columns=cols)



    


