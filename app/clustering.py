import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import List
import datetime
import pickle
from clean import *
from graph import *
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from scipy.spatial.distance import cdist
from dtaidistance import dtw
import scipy.cluster.hierarchy as hac
from scipy import stats
from sklearn.cluster import AgglomerativeClustering


import sys

def loadData(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def testModel(params, X, K, image1_name, image2_name):
    inertias = []
    silhouettes = []
    for k in range(2, K):
        print(f'k: {k}')
        model = TimeSeriesKMeans(metric=params['metric'], n_clusters=k,n_jobs=params['n_jobs'])
        model.fit(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, model.labels_))

    Ks = [*range(2,K)]
    plt.plot(Ks, inertias)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('K Means Inertia Elbow Plot')
    plt.xticks(Ks)
    plt.savefig(image1_name)
    plt.close()

    plt.plot(Ks, silhouettes)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('K Means Silhouette Score Elbow Plot')
    plt.xticks(Ks)
    plt.savefig(image2_name)

def fitAndSaveModel(model, X, file):
    model.fit(X)
    model.to_json(file)

def plotClusters(model, file):
    y_pred = model.labels_
    centroids = model.cluster_centers_
    score = silhouette_score(X, y_pred)
    plt.figure()
    for i, centroid in enumerate(centroids):
        plt.subplot(2,4, i+1)
        clusters = X[y_pred == i]
        cluster_size = len(clusters)
        for ts in clusters: # get each row in the current cluster
            plt.plot(ts)
            plt.text(0.1,0.85,s=f'Cluster {i+1}, size={cluster_size}', fontsize='x-small',transform=plt.gca().transAxes)
        plt.plot(centroid, 'r-')
        if(i == 0):
            plt.title("K-Means DTW Clusters")
    plt.savefig(file, dpi=150)
    return score

def getClusters(columns, labels):
    clusters = {}
    for i, label in enumerate(labels):
        col_name = columns[i]
        clusters[col_name] = label
    return clusters
 
def hierarchicalClustering(distance_matrix, columns,method='Complete'):
    if method == 'Complete':
        Z = hac.complete(distance_matrix)
    if method == 'Single':
        Z = hac.single(distance_matrix)
    if method == 'Average':
        Z = hac.average(distance_matrix)
    if method == 'Ward':
        Z = hac.ward(distance_matrix)
    
    fig = plt.figure(figsize=(16, 8))
    dn = hac.dendrogram(Z, labels=columns)
    plt.title(f"Dendrogram for {method}-Linkage with Dynamic Time Warping Distance")
    plt.ylabel('Distance')
    plt.savefig('dendrogram.png')
    return Z

def getDtwMatrix(X):
    dtw_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i, row1 in enumerate(X):
        row1 = row1.flatten()
        print(i, row1.shape)
        for j, row2 in enumerate(X):
            row2 = row2.flatten()
            print(f'...{j} , {row2.shape}')
            if i != j:
                dist = dtw.distance_fast(row1, row2)
                dtw_matrix[i, j] = dist
    return dtw_matrix

def plotClustersAg(model, title):
    y_pred = model.labels_
    score = silhouette_score(X, y_pred)
    plt.figure()
    labels = np.unique(y_pred)
    for i in labels:
        plt.subplot(2,4, i+1)
        clusters = X[y_pred == i]
        cluster_size = len(clusters)
        for ts in clusters: # get each row in the current cluster
            plt.plot(ts)
            plt.text(0.1,0.85,s=f'Cluster {i+1}, size={cluster_size}', fontsize='x-small',transform=plt.gca().transAxes)
        if(i == 0):
            plt.title(title)
    plt.savefig(f'{title}_clusters.png', dpi=150)
    return score
        
    
graph = loadData('graph_object.pk1')
model = TimeSeriesKMeans.from_json('k_means_dtw.json')
columns = list(graph.datasets.columns)
labels_kmeans = {'epi': 3, 'djia': 1, 'nasdaq': 1, 'russel2000': 1, 's&p500': 1, 'tBond': 1, 'tBill': 1, 'cBond': 1, 'mBond': 1, 'aluminum': 2, 'copper': 2, 'gold': 2, 'lead': 5, 'nickel': 2, 'silver': 7, 'zinc': 1, 'crude': 7, 'natgas': 0, 'petroleum': 7, 'biofuel': 7, 'cattle': 6, 'wheat': 2, 'cocoa': 4, 'coffee': 0, 'corn': 7, 'cotton': 2, 'leanhogs': 6, 'soybeans': 7, 'sugar': 5}
dm = np.load('dtw_matrix.npy')
X = graph.datasets.to_numpy().T
# X = np.array(to_time_series_dataset(X), dtype=np.double)

# normalized=np.empty(X.shape)
# for i,row in enumerate(X):
#     normalized[i] = stats.zscore(row)





params = {"metric": "dtw","n_clusters":8,"n_jobs": -1} # {'metric': 'dtw'}


# distance_matrix_correlation = graph.corr_vec
# distance_matrix_dtw = getDtwMatrix(X)
# distance_matrix_dtw = dtw.distance_matrix_fast(X, compact=True)
#dm = getDtwMatrix(X)

# hierarchicalClustering(dm,columns)
scores = {}
for num_cluster in range(4,14):
    print(num_cluster)
    for linkage in ['complete', 'average']:
        print("..."+linkage)
        cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', connectivity=dm, linkage=linkage) 
        y_pred = cluster.fit_predict(X) 
        scores[f'{linkage}_{num_cluster}'] = silhouette_score(X, y_pred)
# scores['kmeans'] = plotClusters(model, "k-means_clustres.png")
print(scores)







    




