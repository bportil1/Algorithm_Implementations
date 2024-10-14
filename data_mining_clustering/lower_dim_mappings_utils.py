import numpy as np
from time import time
import matplotlib.pyplot as plt

from matplotlib import offsetbox

from sklearn.preprocessing import MinMaxScaler

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

def plot_embedding(X, title):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")

def visualization_tester(X, y):
    n_neighbors = 20
    embeddings = {
        "Random projection embedding": SparseRandomProjection(n_components=2, random_state=42),
        "Truncated SVD embedding": TruncatedSVD(n_components=2),
        #"Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=2),
        #"Standard LLE embedding": LocallyLinearEmbedding(
        #    n_neighbors=n_neighbors, n_components=2, method="standard"
        #),
        #"Modified LLE embedding": LocallyLinearEmbedding(
        #    n_neighbors=n_neighbors, n_components=2, method="modified"
        #),
        #"LTSA LLE embedding": LocallyLinearEmbedding(
        #    n_neighbors=n_neighbors, n_components=2, method="ltsa"
        #),
        #"MDS embedding": MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
        #"Random Trees embedding": make_pipeline(
        #    RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
        #    TruncatedSVD(n_components=2),
        #),
        #"Spectral embedding": SpectralEmbedding(
        #    n_components=2, random_state=0, eigen_solver="arpack"
        #),
        #"t-SNE embedding": TSNE(
        #    n_components=2,
        #    max_iter=500,
        #    n_iter_without_progress=150,
        #    n_jobs=2,
        #    random_state=0,
        #),
        #"NCA embedding": NeighborhoodComponentsAnalysis(
        #    n_components=2, init="pca", random_state=0
        #),
    }

    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        #if name.startswith("Linear Discriminant Analysis"):
        #    data = X.copy()
        #    data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        #else:
        data = X

        print(f"Computing {name}...")
        start_time = time()
        projections[name] = transformer.fit_transform(data, y)
        timing[name] = time() - start_time

    for name in timing:
        title = f"{name} (time {timing[name]:.3f}s)"
        plot_embedding(projections[name], title)

    plt.show()

    return 0



