from spread_opt import *
from preprocessing_utils import *
from clustering import *

if __name__ == '__main__':
    #ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/home/bryanportillo_lt/Documents/py_env/venv/network_intrusion_dataset/Train_data.csv'

    data_obj = data()

    data_obj.load_data(ids_train_file, 'train')

    data_obj.load_labels('train', from_data=True)

    data_obj.split_data(.2)

    data_obj.encode_categorical('protocol_type', 'data')

    data_obj.encode_categorical('service', 'data')

    data_obj.encode_categorical('flag', 'data')

    data_obj.encode_categorical('class', 'labels')
   
    #print(data_obj.train_data.tail(5))
    #print(data_obj.train_labels.tail(5))

    #data_obj.scale_data('min_max')

    init_path = './results/orig_data_visualization/'

    os.makedirs(init_path, exist_ok=True)

    #data_obj.lower_dimensional_embedding(data_obj.train_data, 'train', 'Original Train Data: 3-Dimensions', init_path)

    #data_obj.lower_dimensional_embedding(data_obj.test_data, 'test', 'Original Test Data: 3-Dimensions', init_path)

    data_obj.generate_graphs('train')

    data_obj.generate_graphs('test')

    print(data_obj.train_data.tail(5))
    print(data_obj.train_labels.tail(5))
    print(data_obj.test_data.tail(5))
    print(data_obj.test_labels.tail(5))

    print(data_obj.train_graph)
    print(data_obj.test_graph)

    init_path = './results/plain_data/'

    os.makedirs(init_path, exist_ok=True)
    
    clustering_meths = [           'Kmeans',
                                   'Agglomerative',
                                   'Spectral',
                                   'HDBSCAN',
                                   'MeanShift',
                                   'OPTICS',
                                   'Birch',
                                   'BisectingKmeans'
                                   ]

    #data_obj.lower_dimensional_embedding(data_obj.train_graph ,'train', 'Original Train Graph: 3-Dimensions', init_path)

    #data_obj.lower_dimensional_embedding(data_obj.test_graph, 'test', 'Original Test Graph: 3-Dimensions', init_path)
    
    #plain_clustering = clustering(data_obj.train_data, data_obj.train_labels, data_obj.test_data, data_obj.test_labels, "full", "40_dim_no_proj", clustering_methods=clustering_meths, workers = -1)

    #print(plain_clustering.clustering_methods)
    #print(plain_clustering.clustering_funcs)

    #plain_clustering.generate_clustering()
    
        
    #prec_gamma = np.ones(data_obj.train_data.loc[[0]].shape[1]) * .15 
    '''
    prec_gamma = np.asarray( [0.15,       0.15,       0.08022834, 0.12596744, 0.14999965, 0.14999928,
 0.15,       0.15,       0.15,       0.14995772, 0.15,       0.15,
 0.15,       0.15,       0.15,       0.15,       0.15,       0.15,
 0.1440897,  0.15,       0.15,       0.15,       0.07526867, 0.12220272,
 0.13014209, 0.1322162,  0.1406977,  0.14065416, 0.09846964, 0.1373013,
 0.06716577, 0.07199228, 0.11924177, 0.13586606, 0.14487602, 0.11904342,
 0.14467075, 0.14900498, 0.14499863, 0.135036538, 0.12360249] )

    aew_train = aew(data_obj.train_graph, data_obj.train_data) # , prec_gamma)

    #aew_train.generate_optimal_edge_weights(1000)

    aew_train.generate_edge_weights()

    aew_test = aew(data_obj.test_graph, data_obj.test_data, aew_train.gamma)

    aew_test.generate_edge_weights()

    #print("Train NaNs: ", np.count_nonzero(np.isnan(aew_train.similarity_matrix)))

    #print("Test NaNs: ", np.count_nonzero(np.isnan(aew_test.similarity_matrix)))

    #print("Train Similarity?: ", (aew_train.similarity_matrix==aew_train.similarity_matrix.T).all())

    #print("Test Similarity?: ", (aew_test.similarity_matrix==aew_test.similarity_matrix.T).all())

    #print("Similarity?: ", (aew_train.similarity_matrix==aew_train.similarity_matrix.T).all())

    #print("Similarity?: ", (aew_test.similarity_matrix==aew_test.similarity_matrix.T).all())

    plain_graph_clustering = clustering(aew_train.similarity_matrix, data_obj.train_labels, aew_test.similarity_matrix, data_obj.test_labels, "full", "40_dim_no_proj_graph_data", clustering_methods=clustering_meths,  workers = -1)

    plain_graph_clustering.generate_clustering()
    '''

    clusters = clustering(workers=-1)

    clusters.generate_clustering()


    '''
    num_components = [3, 4, 5, 6, 7, 8, 9, 10] #, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 35, 40]

    for num_comp in num_components:

        #print("Current number of components: ", num_comp)

        data_obj.train_projection, _ = data_obj.downsize_data(aew_train.similarity_matrix, 'train', num_comp)

        data_obj.test_projection, _ = data_obj.downsize_data(aew_test.similarity_matrix, 'test', num_comp)
        init_path = './results/orig_data_visualization/num_comp_' + str(num_comp) + '/'

        os.makedirs(init_path, exist_ok=True)

        for projection in data_obj.train_projection.keys():

            #print("Train NaNs: ", np.count_nonzero(np.isnan(data_obj.train_projection[projection])))    

            #print("Test NaNs: ", np.count_nonzero(np.isnan(data_obj.test_projection[projection])))

            data_obj.lower_dimensional_embedding(data_obj.train_projection[projection], 'train', 'Train Mappings Base: 3-Dimensions', init_path)

            data_obj.lower_dimensional_embedding(data_obj.test_projection[projection], 'test', 'Test Mappings Base: 3-Dimensions', init_path)

            clustering_graph_data = clustering(data_obj.train_projection[projection], data_obj.train_labels, data_obj.test_projection[projection], data_obj.test_labels, num_comp, projection, clustering_methods=clustering_meths, workers = -1)

            clustering_graph_data.generate_clustering()
    '''
