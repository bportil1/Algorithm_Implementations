from spread_opt import *
from preprocessing_utils import *
from clustering import *

from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/home/bryanportillo_lt/Documents/py_env/venv/network_intrusion_dataset/Train_data.csv'
    
    #synth_clust = clustering()

    #synth_clust.synthetic_data_tester()
    
    data_obj = data()

    data_obj.load_data(ids_train_file, 'train')

    data_obj.load_labels('train', from_data=True)

    data_obj.split_data(.5)

    data_obj.encode_categorical('protocol_type', 'data')

    data_obj.encode_categorical('service', 'data')

    data_obj.encode_categorical('flag', 'data')

    data_obj.encode_categorical('class', 'labels')
   
    #print(data_obj.train_data.tail(10))
    #print(data_obj.train_labels.tail(5))

    data_obj.scale_data('min_max')

    #print(data_obj.train_data.tail(10))

    init_path = './results/orig_data_visualization/'

    os.makedirs(init_path, exist_ok=True)

    #data_obj.lower_dimensional_embedding(data_obj.train_data, 'train', 'Original Train Data: 3-Dimensions', init_path)

    #data_obj.lower_dimensional_embedding(data_obj.test_data, 'test', 'Original Test Data: 3-Dimensions', init_path)

    data_obj.generate_graphs('train')

    data_obj.generate_graphs('test')

    init_path = './results/plain_data/'

    os.makedirs(init_path, exist_ok=True)
    
    clustering_meths = [           'Kmeans',
                                   'Agglomerative',
                                   'Spectral',
                                   'Birch',
                                   'BisectingKmeans',
                                   'GaussianMixture'
                                   ]

    data_obj.lower_dimensional_embedding(data_obj.train_graph ,'train', 'Original Train Graph: 3-Dimensions', init_path)

    data_obj.lower_dimensional_embedding(data_obj.test_graph, 'test', 'Original Test Graph: 3-Dimensions', init_path)
    
    #plain_clustering = clustering(data_obj.train_data, data_obj.train_labels, data_obj.test_data, data_obj.test_labels, "full", "40_dim_no_proj", clustering_methods=clustering_meths, workers = -1)

    #plain_clustering.generate_clustering()
        
    #prec_gamma = np.ones(data_obj.train_data.loc[[0]].shape[1]) * .15 
    
    prec_gamma = np.asarray( [ 1.41185215e-01,  7.47973409e+00, -4.13868954e+01,  6.23689652e+00,
.01395224e+01,  2.95441189e+00,  2.02258776e+00,  1.51630823e+00,
   2.02617273e+00, -6.30117218e+01,  1.49669532e+01, -8.95148111e+01,
   1.44078287e+02,  5.37490695e+02,  3.94267329e-01, -1.98542978e+00,
   8.01043655e-01,  8.97132437e-01,  1.46472018e+02,  1.44511124e+00,
   4.11466407e-01,  8.25410226e-01,  6.75843157e-01,  4.33373884e-01,
   9.52821813e-01,  2.98303996e+00, -2.06645769e+02, -2.04183439e+02,
  -2.70176195e+01, -2.49866350e+01,  1.89975904e+02, -1.25376397e+01,
   3.28381552e+01,  1.64834693e+02, -1.39526951e+01,  4.60787859e+01,
   1.06103639e+01, -2.01538991e+02, -2.03414200e+02, -2.13656865e+01,
  -2.26909766e+01]
)

    prec_gamma = np.asarray( [ 3.16832327e-01,  7.27582996e+00, -4.14996719e+01,  6.23296543e+00,
   1.01346893e+01,  2.95074041e+00,  2.02236888e+00,  1.51625221e+00,
   2.02527163e+00, -6.31102784e+01,  1.49306789e+01, -8.96979641e+01,
   1.43775900e+02,  5.34809190e+02,  3.94267329e-01, -1.98693606e+00,
   8.01043655e-01,  8.96859679e-01,  1.46211633e+02,  1.44445228e+00,
   4.11354366e-01,  8.25410226e-01,  6.75621161e-01,  4.33373884e-01,
   9.52821813e-01,  2.97737111e+00, -2.06805308e+02, -2.04341934e+02,
  -2.68355430e+01, -2.48061128e+01,  1.89797745e+02, -1.25752823e+01,
   3.27718486e+01,  1.64492289e+02, -1.38098710e+01,  4.62492982e+01,
   1.05845759e+01, -2.01697391e+02, -2.03567852e+02, -2.11861811e+01,
  -2.25106438e+01])

    prec_gamma = np.asarray( [ 3.24038840e-01,  6.21767301e+00, -4.24643810e+01,  6.23031173e+00,
   1.01307221e+01,  2.93756741e+00,  2.01594072e+00,  1.51188821e+00,
   2.02264289e+00, -6.34205156e+01,  1.48430805e+01, -9.11153330e+01,
   1.42883398e+02,  5.25993603e+02,  3.94267329e-01, -1.99120583e+00,
   8.00879669e-01,  8.96859679e-01,  1.45438132e+02,  1.44428829e+00,
   4.11354366e-01,  8.25410226e-01,  6.74128154e-01,  4.33373884e-01,
   9.52821813e-01,  2.96183387e+00, -2.07395616e+02, -2.04933318e+02,
  -2.70822393e+01, -2.50513445e+01,  1.88504864e+02, -1.27152750e+01,
   3.25829171e+01,  1.63464349e+02, -1.39655886e+01,  4.59896549e+01,
   1.05229022e+01, -2.02291535e+02, -2.04147122e+02, -2.14144698e+01,
  -2.27530551e+01])

    rng = np.random.default_rng()

    #prec_gamma = rng.random(size=(1, 41)) 

    #prec_gamma = np.random.randint(0, 200, (1, 41))

    #prec_gamma = prec_gamma.astype(float)

    prec_gamma = np.var(data_obj.train_data, axis=0).values

    print(prec_gamma)

    

    aew_train = aew(data_obj.train_graph, data_obj.train_data, data_obj.train_labels, prec_gamma)

    aew_train.generate_optimal_edge_weights(5)

    #aew_train.generate_edge_weights()

    prec_gamma = np.var(data_obj.test_data, axis=0).values

    aew_test = aew(data_obj.test_graph, data_obj.test_data, data_obj.test_labels, prec_gamma)

    #print(np.isnan(aew_train.similarity_matrix).any())

    #print(len(aew_train.similarity_matrix[0]))

    #print(len(aew_train.similarity_matrix))

    aew_test.generate_edge_weights()

    #print(np.isnan(aew_test.similarity_matrix).any())

    #print(len(aew_test.similarity_matrix[0]))

    #print(len(aew_test.similarity_matrix))

    
    clustering_with_adj_matr_prec_kmeans = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)

    print("Kmeans Train: ", accuracy_score(clustering_with_adj_matr_prec_kmeans.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_disc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=-1)

    print("Discretize Train: ", accuracy_score(clustering_with_adj_matr_prec_disc.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_clust = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='cluster_qr', n_jobs=-1)

    print("Cluster_qr Train: ", accuracy_score(clustering_with_adj_matr_prec_clust.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_kmeans1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)

    print("Kmeans Test: ", accuracy_score(clustering_with_adj_matr_prec_kmeans1.fit_predict(aew_test.eigenvectors), aew_test.labels))

    clustering_with_adj_matr_prec_disc1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=-1)

    print("Discretize Test: ", accuracy_score(clustering_with_adj_matr_prec_disc1.fit_predict(aew_test.eigenvectors), aew_test.labels))

    clustering_with_adj_matr_prec_clust1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='cluster_qr', n_jobs=-1)

    print("Cluster_qr Test: ", accuracy_score(clustering_with_adj_matr_prec_clust1.fit_predict(aew_test.eigenvectors), aew_test.labels))
    
    
    plain_graph_clustering = clustering(aew_train.eigenvectors, aew_train.labels, aew_test.eigenvectors, aew_test.labels, "full", "40_dim_no_proj_graph_data", clustering_methods=clustering_meths,  workers = -1)

    plain_graph_clustering.generate_clustering()

'''
    num_components = [3, 8, 12, 15, 20, 40]

    for num_comp in num_components:

        print("Current number of components: ", num_comp)

        data_obj.train_projection, _ = data_obj.downsize_data(aew_train.eigenvectors, 'train', num_comp)

        data_obj.test_projection, _ = data_obj.downsize_data(aew_test.eigenvectors, 'test', num_comp)
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
