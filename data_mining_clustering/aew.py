from spread_opt import *
from preprocessing_utils import *
from clustering import *

if __name__ == '__main__':
    ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/home/bryanportillo_lt/Documents/py_env/venv/network_intrusion_dataset/Train_data.csv'

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

    data_obj.lower_dimensional_embedding(data_obj.train_data, 'train', 'Original Train Data: 3-Dimensions', init_path)

    data_obj.lower_dimensional_embedding(data_obj.test_data, 'test', 'Original Test Data: 3-Dimensions', init_path)

    data_obj.generate_graphs('train')

    data_obj.generate_graphs('test')

    init_path = './results/plain_data/'

    os.makedirs(init_path, exist_ok=True)
    '''
    data_obj.lower_dimensional_embedding(data_obj.train_graph ,'train', 'Original Train Graph: 3-Dimensions', init_path)

    data_obj.lower_dimensional_embedding(data_obj.test_graph, 'test', 'Original Test Graph: 3-Dimensions', init_path)
    
    plain_clustering = clustering(data_obj.train_data, data_obj.train_labels, data_obj.test_data, data_obj.test_labels, "40_dim_no_proj", workers = -1)

    plain_clustering.generate_clustering()
    '''
    prec_gamma = np.ones(data_obj.train_data.loc[[0]].shape[1]) * .15 

    aew_train = aew(data_obj.train_graph, data_obj.train_data, prec_gamma)

    aew_train.generate_optimal_edge_weights(10)

    aew_test = aew(data_obj.test_graph, data_obj.test_data, aew_train.gamma)

    aew_test.generate_edge_weights()

    num_components = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 35, 40]

    for num_comp in num_components:

        print("Current number of components: ", num_comp)

        data_obj.train_projection, _ = data_obj.downsize_data(data_obj.train_graph, 'train', num_comp)

        data_obj.test_projection, _ = data_obj.downsize_data(data_obj.test_graph, 'test', num_comp)
        init_path = './results/orig_data_visualization/num_comp_' + str(num_comp) + '/'

        os.makedirs(init_path, exist_ok=True)

        for projection in data_obj.train_projection.keys():

            data_obj.lower_dimensional_embedding(data_obj.train_projection[projection], 'train', 'Train Mappings Base: 3-Dimensions', init_path)

            data_obj.lower_dimensional_embedding(data_obj.test_projection[projection], 'test', 'Test Mappings Base: 3-Dimensions', init_path)

            clustering = clustering(data_obj.train_projection[projection], data_obj.train_labels, data_obj.test_projection[projection], data_obj.test_labels, workers = -1)

            clustering.generate_clustering()
