from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

from sklearn.cluster import KMeans

def fit_kmeans(X,k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    return kmeans.inertia_
    #return None


def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    blob_data, _ = make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12)
    mb_1=blob_data[0:,0:1]
    mb_2=blob_data[0:,1:]
    dct = answers["2A: blob"] = [mb_1,mb_2,_]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    result = dct

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair

    sse_values = []
    k_range = range(1, 9)
    for k in k_range:
        sse = fit_kmeans(blob_data, k)
        sse_values.append([k, sse])

    dct = answers["2C: SSE plot"] = sse_values
    def plot_sse(sse_values):
    # Unzip the list of tuples into k and SSE values
        k_values, sse = zip(*sse_values)

    # Plot SSE as a function of k
        plt.plot(k_values, sse, marker='o', linestyle='-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('SSE as a function of k')
        plt.xticks(k_values)
        plt.grid(True)
        plt.show()
    # Call the function to plot SSE
    plot_sse(sse_values)

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    inertia_values = []
    k_range = range(1,9)
    for k in k_range:
        inertia = fit_kmeans(blob_data, k)
        inertia_values.append([k, inertia])

    answers["2D: inertia plot"] = inertia_values
    '''
    # Determine if optimal k's agree
    optimal_k_sse = min(sse_values, key=lambda x: x[1])[0]
    optimal_k_inertia = min(inertia_values, key=lambda x: x[1])[0]
    do_ks_agree = "yes" if optimal_k_sse == optimal_k_inertia else "no"
    
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = do_ks_agree
    '''
    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
