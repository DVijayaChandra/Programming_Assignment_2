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
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    # import plotly.figure_factory as ff
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn import datasets
    import math
    from sklearn.cluster import AgglomerativeClustering
    import pickle
    import utils as u

    """
    Part 4.	
    Evaluation of Hierarchical Clustering over Diverse Datasets:
    In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
    """

    # Fill these two functions with code at this location. Do NOT move it. 
    # Change the arguments and return according to 
    # the question asked. 


    def fit_hierarchical_cluster(dataset, n_clusters, linkage='ward'):
        data, _ = dataset  # Labels are not used in clustering

        # Scale the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Perform Agglomerative Clustering
        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster.fit(data_scaled)

        return cluster.labels_

    def fit_modified(dataset, distance_threshold, linkage_method):
        # Extract data, ignoring labels since clustering is unsupervised
        data, _ = dataset

        # Scale the data for uniformity in distance calculation
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Set up and fit the hierarchical clustering model
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage_method)
        model.fit(data_scaled)

        # Return the cluster labels assigned to each data point
        return model.labels_


    def compute():
        answers = {}

        """
        A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
        """

        # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
        # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
        dct = answers["4A: datasets"] = {}
        n_samples = 100
        seed = 42

        nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
        dct['nc'] = list(nc)
        nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        dct['nm'] = list(nm)
        bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
        dct['bvv'] = list(bvv)
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        add = (X_aniso, y)
        dct['add'] = list(add)
        b = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        dct['b'] = list(b)
        # dct value:  the `fit_hierarchical_cluster` function
        dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

        """
        B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

        Create a pdf of the plots and return in your report. 
        """
        n_samples = 100
        random_state = 42

        # Generate datasets
        nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
        nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
        bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        add = (X_aniso, y)
        b = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

        given_datasets = {
            "nc": nc,
            "nm": nm,
            "bvv": bvv,
            "add": add,
            "b": b
        }

        # Function to fit hierarchical clustering
        def fit_hierarchical_cluster_linkage(data, n_clusters, linkage):
            scaler = StandardScaler()
            # Ensure data is correctly shaped: 2D array where rows are samples
            data_scaled = scaler.fit_transform(data)  # Correctly pass the whole 2D dataset

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            model.fit(data_scaled)

            return model.labels_


        # Prepare for plotting
        linkage_types = ['single', 'complete', 'ward', 'average']
        dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
        pdf_filename = "report_4B.pdf"

        fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16), squeeze=False)
        fig.suptitle('Scatter plots for different datasets and linkage types (2 clusters)', fontsize=16)

        for i, linkage_type in enumerate(linkage_types):
            for j, dataset_key in enumerate(dataset_keys):
                data, labels = given_datasets[dataset_key]
                predicted_labels = fit_hierarchical_cluster_linkage(data, n_clusters=2, linkage=linkage_type)
                ax = axes[i][j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}, k=2')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure to a PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        # dct value: list of dataset abbreviations (see 1.C)
        dct = answers["4B: cluster successes"] = ["nc","nm"]

        """
        C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.

        Create a pdf of the plots and return in your report. 
        """
        def calculate_distance_threshold(data, linkage_type):
            # Make sure the data is in the correct shape (2D)
            if data.ndim == 1:
                data = data.reshape(-1, 1)  # Reshape if data is 1D

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            Z = linkage(data_scaled, method=linkage_type)
            merge_distances = np.diff(Z[:, 2])
            max_rate_change_idx = np.argmax(merge_distances)

            distance_threshold = Z[max_rate_change_idx, 2]
            return distance_threshold


        # Modified version of the clustering function to use distance threshold
        def fit_modified(data, distance_threshold, linkage_method):
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            Z = linkage(data_scaled, method=linkage_method)

            # Using the fcluster function to form flat clusters from the linkage matrix
            predicted_labels = fcluster(Z, distance_threshold, criterion='distance')
            return predicted_labels

        # Generate example datasets
        n_samples = 100
        random_state = 42
        nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
        nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
        bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        add = (X_aniso, y)
        b = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

        given_datasets = {"nc": nc, "nm": nm, "bvv": bvv, "add": add, "b": b}
        dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
        linkage_types = ['single', 'complete', 'ward', 'average']
        pdf_filename = "report_4C.pdf"

        # Preparing for plotting
        fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16), squeeze=False)
        fig.suptitle('Scatter plots for different datasets and linkage types', fontsize=16)

        for i, linkage_type in enumerate(linkage_types):
            for j, dataset_key in enumerate(dataset_keys):
                data, _ = given_datasets[dataset_key]
        
                # Ensure data is in 2D shape; this assumes 'data' is already 2D.
                distance_threshold = calculate_distance_threshold(data, linkage_type)
        
                # If 'data' was modified to ensure 2D shape inside 'calculate_distance_threshold',
                # it should be directly compatible with 'fit_modified' without further changes.
                predicted_labels = fit_modified(data, distance_threshold, linkage_type)
        
                ax = axes[i][j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Saving the figure to a PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

        # dct is the function described above in 4.C
        dct = answers["4A: modified function"] = fit_modified

        return answers


    # ----------------------------------------------------------------------
    if __name__ == "__main__":
        answers = compute()

        with open("part4.pkl", "wb") as f:
            pickle.dump(answers, f)
