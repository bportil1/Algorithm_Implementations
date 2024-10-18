import numpy as np
from time import time
import matplotlib.pyplot as plt

import pandas as pd

from matplotlib import offsetbox

import plotly.express as px

from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

def plot_ids_embedding(X, labels, title):
    X = MinMaxScaler().fit_transform(X)
    #X = normalize(X) 
    
    cdict = {'normal': 'blue', 'anomaly': 'red'}

    x1 = np.asarray([ x[0] for x in X])

    x2 = np.asarray([ x[1] for x in X])

    x3 = np.asarray([ x[2] for x in X])

    df = pd.DataFrame({ 'x1': x1,
                        'x2': x2,
                        'x3': x3,
                        'label': labels })
    #_,  ax = plt.subplots()
    
    for g in np.unique(labels):
        idx = np.where(labels == g)

        fig = px.scatter_3d(df, x='x1', y='x2', z='x3', 
                            color='label', color_discrete_map=cdict,
                            opacity=.4)

        fig.update_layout(
            title = title
        )
    fig.show()


def visualization_tester(X, y):
    n_neighbors = 20
    #standard lle dying sometimes because of a singular value matrix?, swapping eigensolver to dense upped the computation time substantially but ran on initial trial
    embeddings = {
        "Truncated SVD embedding": TruncatedSVD(n_components=3),
        #"Standard LLE embedding": LocallyLinearEmbedding(
        #    n_neighbors=n_neighbors, n_components=3, method="standard", 
        #    eigen_solver='dense', n_jobs=-1
        #),
        "Random Trees embedding": make_pipeline(
            RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0, n_jobs=-1),
            TruncatedSVD(n_components=3),
        ),
        #"t-SNE embedding": TSNE(
        #    n_components=3,
        #    max_iter=500,
        #    n_iter_without_progress=150,
        #    n_jobs=-1,
        #    random_state=0,
        #),
    }

    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        print(f"Computing {name}...")
        start_time = time()
        projections[name] = transformer.fit_transform(X, y)
        timing[name] = time() - start_time

    for name in timing:
        title = f"{name} (time {timing[name]:.3f}s)"
        plot_ids_embedding(projections[name], y, title)
    return
