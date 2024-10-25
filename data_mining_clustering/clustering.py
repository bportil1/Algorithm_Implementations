import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.semi_supervised import LabelPropagation
from sklearn.cluster import HDBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import BisectingKMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from preprocessing_utils import *


class clustering():
    def __init__(self, train_data, train_labels, test_data, test_labels, projection_type, clustering_methods=None, workers = 1):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.base_path = './results/'
        self.clustering_methods = clustering_methods
        self.clustering_funcs = None
        self.projection_type = projection_type
        self.workers = workers
        
        if clustering_methods == None:
            self.get_clustering_methods()
    
        self.get_clustering_funcs()

    def get_clustering_methods(self):
        self.clustering_methods = ['Kmeans',
                                   'Spectral', 
                                   'Agglomerative',
                                   #'LabelProp',
                                   #'DBSCAN', 
                                   'HDBSCAN', 
                                   'MeanShift',
                                   'OPTICS', 
                                   'Birch', 
                                   'BisectingKmeans'
                                   ]

    def get_clustering_funcs(self):

        self.clustering_funcs = {
                'Kmeans': self.generate_kmeans(self.get_clustering_hyperparams('Kmeans')),
                'Spectral': self.generate_spectral(self.get_clustering_hyperparams('Spectral')),
                'Agglomerative': self.generate_agglomerative(self.get_clustering_hyperparams('Agglomerative')),
                #'LabelProp': self.generate_labelprop(self.get_clustering_hyperparams('LabelProp')),
                #'DBSCAN': self.generate_dbscan(self.get_clustering_hyperparams('DBSCAN')),
                'HDBSCAN': self.generate_hdbscan(self.get_clustering_hyperparams('HDBSCAN')),
                'MeanShift': self.generate_meanshift(self.get_clustering_hyperparams('MeanShift')),
                'OPTICS': self.generate_optics(self.get_clustering_hyperparams('OPTICS')),
                'Birch': self.generate_birch(self.get_clustering_hyperparams('Birch')),
                'BisectingKmeans': self.generate_bisectingkmeans(self.get_clustering_hyperparams('BisectingKmeans')),

            }
 
    def get_clustering_hyperparams(self, cluster_alg):

        clustering_params = {
                'Kmeans': {'k_means_alg': ('lloyd', 'elkan'),
                           'n_clusters' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
                },
                'Spectral': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                             'assign_labels': ('kmeans', 'discretize', 'cluster_qr'),
                             'workers': self.workers
                },
                'Agglomerative': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                                  'linkage': ('ward', 'complete', 'average', 'single'),
                                  'metric': ('euclidean', 'manhattan')
                },
                'LabelProp': {'kernel': ('rbf', 'knn'),
                              'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                              'gamma': [15, 18, 20, 23, 25],
                              'workers': self.workers
                },
                #'DBSCAN': {'eps': np.arange(5,150 , step = 5),
                #           'min_samples': [5, 10, 15, 20, 25, 30]   
                #},
                'HDBSCAN': {"min_cluster_size": [5, 10, 15, 20, 25, 30]
                },
                'MeanShift': {'workers': self.workers
                },
                'OPTICS': {'min_samples': [5, 10, 15, 20, 25, 30]
                },
                'Birch': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                          'threshold': [.03, .05, .07]
                },
                'BisectingKmeans': {'k_means_alg': ('lloyd', 'elkan'),
                                    'n_clusters' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                                    'bisecting_strategy': ('biggest_inertia', 'largest_cluster')
                },
        }
        return clustering_params[cluster_alg]

    def generate_clustering(self):
        for alg in self.clustering_methods:
            ctg_matrices_path = self.base_path + alg + '/ctg_matrices'  
            visualizations_path = self.base_path + alg + '/visualizations' 
            results_path = self.base_path + alg + '/results' 
     
            os.makedirs(ctg_matrices_path, exist_ok=True)
            os.makedirs(visualizations_path, exist_ok=True)
            os.makedirs(results_path, exist_ok=True)

            clustering = self.clustering_funcs[alg]

    def generate_kmeans(self, hyperparams):
        print("Computing Kmeans")
        outpath = self.base_path + "kmeans/"
        for alg in hyperparams['k_means_alg']:
            for num_clust in hyperparams['n_clusters']:
                clustering = KMeans(n_clusters=num_clust, algorithm=alg)
                self.cluster_evaluation('kmeans', (alg, num_clust), clustering) 

    def generate_spectral(self, hyperparams):
        print("Computing Spectral")
        outpath = self.base_path + "spectral/"
        for alg in hyperparams['assign_labels']:
            for num_clust in hyperparams['n_clusters']:
                clustering = SpectralClustering(n_clusters=num_clust, assign_labels=alg, n_jobs=hyperparams['workers'])
                self.cluster_evaluation('spectral', (alg, num_clust), clustering) 

    def generate_agglomerative(self, hyperparams):
        print("Computing Agglomerative")
        outpath = self.base_path + "agglomerative/"
        for alg in hyperparams['linkage']:
            for metric in hyperparams['metric']:
                if alg == 'ward' and metric != 'euclidean':
                    continue
                
                for num_clust in hyperparams['n_clusters']:
                    clustering = AgglomerativeClustering(n_clusters=num_clust, metric=metric, linkage=alg)
                    self.cluster_evaluation('agglomerative', (alg, metric, num_clust), clustering)

    def generate_labelprop(self, hyperparams):
        print("Computing Labelprop")
        outpath = self.base_path + "labelprop/"
        for kernel in hyperparams['kernel']:
            if kernel ==  'rbf':
                for gamma in hyperparams['gamma']:
                    clustering = LabelPropagation(kernel=kernel, gamma=gamma, n_jobs=hyperparams['workers'])
                    self.cluster_evaluation('labelprop', (kernel, gamma), clustering)

            elif kernel == 'knn':
                for num_neighs in hyperparams['n_neighbors']:
                    clustering = LabelPropagation(kernel=kernel, n_neighbors=num_neighs, n_jobs=hyperparams['workers'])
                    self.cluster_evaluation('labelprop', (kernel, num_neighs), clustering)

    def generate_dbscan(self, hyperparams):
        print("Computing DBSCAN")
        outpath = self.base_path + "dbscan/"
        for min_samps in hyperparams['min_samples']:
            for eps in hyperparams['eps']:
                clustering = DBSCAN(eps=eps/100, min_samples=min_samps)
                self.cluster_evaluation('dbscan', (eps, min_samps), clustering) 

    def generate_hdbscan(self, hyperparams):
        print("Computing HDBSCAN")
        outpath = self.base_path + "hdbscan/"
        for min_size in hyperparams['min_cluster_size']:
            clustering = HDBSCAN(min_cluster_size=min_size)
            self.cluster_evaluation('hdbscan', [min_size], clustering)         

    def generate_meanshift(self, hyperparams):
        print("Computing MeanShift")
        outpath = self.base_path + "meanshift/"
        clustering = MeanShift(n_jobs=hyperparams['workers'])
        self.cluster_evaluation('meanshift', ["no params"], clustering)  

    def generate_optics(self, hyperparams):
        print("Computing OPTICS")
        outpath = self.base_path + "optics/"
        for min_samps in hyperparams['min_samples']:
            clustering = OPTICS(min_samples=min_samps)
            self.cluster_evaluation('optics', [min_samps], clustering)

    def generate_birch(self, hyperparams):
        print("Computing Birch")
        outpath = self.base_path + "birch/"
        for thresh in hyperparams['threshold']:
            for n_clusts in hyperparams['n_clusters']:
                clustering = Birch(threshold=thresh, n_clusters=n_clusts)
                self.cluster_evaluation('birch', (n_clusts, thresh), clustering) 

    def generate_bisectingkmeans(self, hyperparams):
        print("Computing BisectingKmeans")
        outpath = self.base_path + "bisectingkmeans/"
        for alg in hyperparams['k_means_alg']:
            for strat in hyperparams['bisecting_strategy']:
                for n_clusts in hyperparams['n_clusters']:
                    clustering = BisectingKMeans(n_clusters=n_clusts, algorithm=alg, bisecting_strategy=strat)
                    self.cluster_evaluation('bisectingkmeans', (n_clusts, alg, strat), clustering)   

    def cluster_evaluation(self, alg, hyperparameters, model):

        ctg_matrices_path = self.base_path + alg + '/ctg_matrices'  
        visualizations_path = self.base_path + alg + '/visualizations'
        results_path = self.base_path + alg + '/results'  

        os.makedirs(ctg_matrices_path, exist_ok=True)
        os.makedirs(visualizations_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        if alg in ('spectral', 'agglomerative', 'dbscan',
                   'hdbscan', 'meanshift', 'optics',
                   'birch'):
            labels = model.fit_predict(self.train_data)
            avg_train_acc = accuracy_score(labels, self.train_labels)
            avg_test_acc = 'null'
            avg_fit_time = 'null'
            labels_pred = model.fit_predict(self.test_data)
        elif alg == 'labelprop':
            model.fit(self.train_data, self.train_labels)
            labels = model.predict(self.train_data)
            avg_train_acc = accuracy_score(labels, self.train_labels)
            avg_test_acc = 'null'
            avg_fit_time = 'null'
            labels_pred = model.predict(self.test_data)
        else:
            cv_res = cross_validate(model, self.train_data, self.train_labels, cv = 5, return_train_score = True)
            avg_train_acc = np.average(cv_res['train_score'])
            avg_test_acc = np.average(cv_res['test_score'])
            avg_fit_time = np.average(cv_res['fit_time'])
            labels_pred = model.fit_predict(self.test_data)
            labels = model.fit_predict(self.train_data)

        labels_true = self.test_labels.values.flatten()

        test_set_acc = accuracy_score(labels_pred, labels_true)

        silhoutte = metrics.silhouette_score(self.train_data, labels, metric='euclidean')     # Silhouette Coefficient
        calinski_harabasz = metrics.calinski_harabasz_score(self.train_data, labels)  # Calinski-Harabasz Index
        davies_bouldin = metrics.davies_bouldin_score(self.train_data, labels)    # Davies-Bouldin Index

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

        d = { 'clustering': alg, 'hyperparameters': hyperparameters, 'train_score_avg': avg_train_acc, 
                'test_score_avg_cv': avg_test_acc, 'avg_fit_time': avg_fit_time, 'test_set_acc': test_set_acc, 'Silhoutte' : silhoutte, 'Calinski_Harbasz' : calinski_harabasz, 'Davies_Bouldin' : davies_bouldin,
             'RAND' : ri , 'ARAND': ari, 'MIS' : mis, 'AMIS' : amis, 'NMIS' : nmis, 'Hmg' : hmg, 'Cmplt' : cmplt,
             'V_meas' : v_meas, 'FMs' : fowlkes_mallows}

        df = pd.DataFrame.from_dict(d)
        
        filename_base = '/alg'

        for hyper_param in hyperparameters:

            filename_base += "_" + str(hyper_param) 

        cntg_mtx_name = ctg_matrices_path + filename_base + ".csv"

        f = open(cntg_mtx_name, 'a')

        f.close()

        np.savetxt(cntg_mtx_name, cntg_mtx, delimiter=',', fmt='%d')

        results_file_name = results_path + filename_base + "_" + self.projection_type  +".csv"

        f = open(results_file_name, 'a')

        f.close()

        df.to_csv(results_file_name, index=False)

        vis_file_name = visualizations_path + filename_base + ".html"

        labels_pred = pd.DataFrame(labels_pred, columns=['class'])

        data_obj = data(test_labels=labels_pred)

        data_obj.lower_dimensional_embedding(self.test_data, 'test', filename_base, vis_file_name)

