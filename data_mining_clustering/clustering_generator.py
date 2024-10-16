def generate_DBSCAN(vector, labels, eps, min_samples, twoD_vector):
    _, ndim = vector.shape

    # Get results for DBSCAN clustering algorithm
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vector)
    # Get DBSCAN cluster labels
    cluster_DBSCAN_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_DBSCAN_Labels)) - (1 if -1 in cluster_DBSCAN_Labels else 0)
    n_noise_ = list(cluster_DBSCAN_Labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, cluster_DBSCAN_Labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, cluster_DBSCAN_Labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, cluster_DBSCAN_Labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels, cluster_DBSCAN_Labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels, cluster_DBSCAN_Labels)
    )
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(vector, cluster_DBSCAN_Labels))
    # confusion_matrix()

    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(cluster_DBSCAN_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_DBSCAN_Labels == k

        xy = twoD_vector[class_member_mask ]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

    plt.title('DBSCAN with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n eps = '+str(eps))
    fig = plt.gcf()
    plt.clf()
    return clustering, fig

def generate_AgglomerativeCluster(vector, labels, twoD_vector, n_cluster =2 , linkage = 'ward' ):

    _, ndim = vector.shape
    clustering = AgglomerativeClustering(n_clusters= n_cluster, linkage= linkage).fit(vector)

    # Get cluster labels
    cluster_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(cluster_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_Labels == k

        xy = twoD_vector[class_member_mask ]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=7,
        )


    plt.title('Agglomerative with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n #clusters = ' + str(n_cluster) + ', linkage = ' + linkage)
    fig = plt.gcf()
    plt.clf()
    return clustering, fig

def generate_spectral_clustering(vector, labels, twoD_vector, n_cluster = 2, assign_labels = 'discretize' ):
    _, ndim = vector.shape
    clustering = SpectralClustering(n_clusters= n_cluster, assign_labels = assign_labels, random_state=0).fit(vector)

    # Get cluster labels
    cluster_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result


    # Black removed and is used for noise instead.
    unique_labels = set(cluster_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_Labels == k

        xy = twoD_vector[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    plt.title('Spectral with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n #clusters = ' + str(n_cluster) + ', labels = ' + assign_labels)
    fig = plt.gcf()
    plt.clf()
    return clustering, fig

def generate_kmeans_clustering(vector, labels, twoD_vector, n_cluster = 2, random_state = 0):
    _, ndim = vector.shape
    clustering = KMeans(n_clusters=n_cluster, random_state = random_state).fit(vector)

    # Get cluster labels
    cluster_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result

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

        labels[str(k + 1)] = xy[0]

        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
            alpha=0.4,
        )

    for each in labels.keys():
        plt.annotate(each, labels[each], weight='bold', size=20)

    plt.title('Kmeans with dim ( ' + str(ndim) + ')\n #clusters = ' + str(n_cluster) )
    fig = plt.gcf()
    plt.show()
    plt.clf()

    return clustering, fig

