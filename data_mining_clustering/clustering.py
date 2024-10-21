import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

class clustering():
    def __init__(self, train_data, train_labels, test_data, test_labels, data_path):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.data_path = data_path
        self.clustering_methods = ['Kmeans', 'spectral', 'Agglomerative', 'DBSCAN']

    def get_clustering_hyperparams(self, cluster_alg):

        if cluster_alg == 'Kmeans':
            hyper_para_name = 'n_clusters'
            random_state = 0 
            hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        elif cluster_alg == 'spectral':
            hyper_para_name = 'n_clusters'
            assign_labels = 'discretize'
            hyper_para_list = np.arange(2, 31, step = 1)
        elif cluster_alg == 'Agglomerative':
            hyper_para_name = 'n_clusters'
            linkage = 'ward'
            hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        elif cluster_alg == 'DBSCAN':
            hyper_para_name = 'eps'
            min_samples = 2
            hyper_para_list = np.arange(5,150 , step = 5)

         ##### turn this line into  class attributes
        return hyper_para_name, hyper_para_list


    def clustering_training(self, save_model = True):

        model_path = self.data_path

        cluster_valuation = pd.DataFrame()

        for cluster_alg in self.clustering_methods:
        
            print("Computing ", cluster_alg)

            hyper_para_name, hyper_para_list = self.get_clustering_hyperparams(cluster_alg)

            for hyper_para in hyper_para_list:

                if cluster_alg == 'Kmeans': 

                    clustering_model = KMeans(n_clusters=hyper_para).fit(self.train_data)
            
                elif cluster_alg == 'spectral':

                    clustering_model = SpectralClustering( n_clusters = hyper_para, affinity='precomputed_nearest_neighbors', assign_labels = 'discretize').fit(self.train_data)


                elif cluster_alg == 'Aggloromative':

                    clustering_model = AgglomerativeClustering(n_clusters= hyper_para, linkage= 'ward').fit(self.train_data)

                elif cluster_alg == 'DBSCAN':

                    clustering_model = DBSCAN(eps=hyper_para/100, min_samples=2).fit(self.train_data)


                if cluster_alg == 'Kmeans':

                    test_pred = clustering_model.predict(self.test_data)

                else:

                    test_pred = clustering_model.fit_predict(self.train_data)

                eval, cntg = self.cluster_evaluation(test_pred, self.test_labels)

                print("Accuracy: ", accuracy_score(test_pred, self.test_labels))

            cluster_valuation = pd.concat([cluster_valuation, eval], ignore_index=True)
            
            cluster_valuation.reset_index()

            if save_model:
            
                cluster_model_path = model_path + '/' + cluster_alg + '/cluster_models'

                os.makedirs(cluster_model_path, exist_ok=True)

                cluster_model_name = (cluster_model_path + '/clustering_model_' + str(hyper_para) + '-clusters.sav')
                
                pickle.dump(clustering_model, open(cluster_model_name, 'wb'))
        
            #cluster_valuation.insert(0, hyper_para_name, hyper_para_list)

            valuation_name = Path(model_path + '/' + cluster_alg + '/test/' + 'test_clustering_evaluation.csv')
            
            valuation_name.parent.mkdir(parents=True, exist_ok=True)
            
            cluster_valuation.to_csv(valuation_name)

    def cluster_evaluation(self, labels_pred, labels_true):

        ri = metrics.rand_score(labels_true, labels_pred)   # RAND score
        ari = metrics.adjusted_rand_score(labels_true, labels_pred) # Adjusted RAND score

        mis = metrics.mutual_info_score(labels_true, labels_pred)  # mutual info score
        amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)    # adjusted mutual information score
        nmis = metrics.normalized_mutual_info_score(labels_true, labels_pred)  # normalized mutual info score

        hmg = metrics.homogeneity_score(labels_true, labels_pred)   # homogeneity
        cmplt = metrics.completeness_score(labels_true, labels_pred)    # completeness
        v_meas = metrics.v_measure_score(labels_true, labels_pred)   # v_measure score

        fowlkes_mallows = metrics.fowlkes_mallows_score(labels_true, labels_pred)   # Fowlkes-Mallows scores

        cntg_mtx = contingency_matrix(labels_true, labels_pred)     # Contingency Matrix

        d = {'RAND' : ri , 'ARAND': ari, 'MIS' : mis, 'AMIS' : amis, 'NMIS' : nmis, 'Hmg' : hmg, 'Cmplt' : cmplt,
                     'V_meas' : v_meas, 'FMs' : fowlkes_mallows}
        df = pd.DataFrame.from_dict(d,  orient='index')
        
        #turn this into class attributes
        return df, cntg_mtx

    def model_evaluation(self):

        silhoutte = metrics.silhouette_score(X, labels, metric='euclidean')     # Silhouette Coefficient
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels)  # Calinski-Harabasz Index
        davies_bouldin = metrics.davies_bouldin_score(X, labels)    # Davies-Bouldin Index

        d = {'Silhoutte' : silhoutte, 'Calinski_Harbasz' : calinski_harabasz, 'Davies_Bouldin' : davies_bouldin}
    
        return d

    def plot_clusters(twoD_vector, cluster_Labels, alg_name ='', hyper_para_name='default', hyp_para= 0):
        core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
        # core_samples_mask[clustering.core_sample_indices_] = True

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
        n_noise_ = list(cluster_Labels).count(-1)
        # 2d plot, may not keep Plot result

        '''
        # Black removed and is used for noise instead.
        unique_labels = set(cluster_Labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        labels = {}

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = cluster_Labels == k

            xy = twoD_vector[class_member_mask]

            labels[str(k+1)] = xy[0]

            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=5,
                alpha= 0.4,
            )
        '''
        for each in labels.keys():
             plt.annotate(each,labels[each], weight= 'bold', size = 20)

        plt.title( alg_name + ' with dim ( ' + str(ndims) + ')\n #'+ hyper_para_name +' = ' + str(hyp_para) )
        fig = plt.gcf()
        plt.show()
        # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
        # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
        # fig.savefig(fname)
        # plt.clf()
        return fig



