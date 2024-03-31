import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from matplotlib.backends.backend_pdf import PdfPages
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
    # Unpack dataset
    data, labels = dataset

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Initialize KMeans estimator
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='random')

    # Fit KMeans to the standardized data
    kmeans.fit(scaled_data)

    # Get predicted cluster labels
    predicted_labels = kmeans.labels_

    return predicted_labels


def compute():
    answers = {}
    dct = answers["1A: datasets"] = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    noisy_circles = datasets.make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    dct['nc'] = noisy_circles
    noisy_moons = datasets.make_moons(n_samples=100, noise=0.05, random_state=42)
    dct['nm'] = noisy_moons
    varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    dct['bvv'] = varied
    X, y = datasets.make_blobs(n_samples=100, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    dct['add'] = aniso
    blobs = datasets.make_blobs(n_samples=100, random_state=42)
    dct['b'] = blobs
    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """


    # Assuming datasets and num_clusters are defined as described in part 1.A
    dataset_key = ['nc','nm','bvv','add','b']  # Assuming aniso is the dataset from part 1.A
    num_clusters = [2, 3, 5, 10]
    pdf_pages = PdfPages("report.pdf")

    # Create a big figure
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))

    # Iterate through each dataset
    for i, dataset in enumerate(dataset_key):
        X, y = dct[dataset]
        # Iterate through each number of clusters
        for j, k in enumerate(num_clusters):
            # Fit KMeans and get predicted labels
            kmeans = KMeans(n_clusters=k, random_state=0)
            predicted_labels = kmeans.fit_predict(X)
            # Plot the scatter plot
            ax = axs[j, i]
            ax.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
            ax.set_title(f'Dataset {i+1}, k={k}')
    pdf_pages.savefig(fig)

    plt.close()  # Close the figure to release resources




    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"xy": [3,4], "zx": [2]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["xy"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = [""]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)

