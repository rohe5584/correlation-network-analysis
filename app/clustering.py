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
        plt.subplot(2,3, i+1)
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
 
def hierarchicalClustering(distance_matrix, columns,method='Ward'):
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
    plt.title(f"Dendrogram for {method}-Linkage with Distance")
    plt.ylabel('Distance')
    plt.savefig('dendrogram_corr.png')
    return Z

def plot_dendrogram(model, columns):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    fig = plt.figure(figsize=(16, 8))
    hac.dendrogram(linkage_matrix, labels=columns)
    plt.title(f"Dendrogram for Ward-Linkage with Euclidean Distance")
    plt.ylabel('Euclidean Distance')
    plt.savefig('dendrogram_ward')

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
        plt.subplot(3,4, i+1)
        clusters = X[y_pred == i]
        cluster_size = len(clusters)
        for ts in clusters: # get each row in the current cluster
            plt.plot(ts)
            plt.text(0.1,0.85,s=f'Cluster {i+1}, size={cluster_size}', fontsize='x-small',transform=plt.gca().transAxes)
        if(i == 0):
            plt.title("Ward Linkage Clusters")
    plt.savefig(f'{title}_clusters.png', dpi=150)
    return score


def plotParams(scores):
    keys = ['ward', 'single', 'complete', 'average']

    ward = []
    single = []
    complete = []
    average = []

    for s in scores:
        if "ward" in s:
            ward.append(scores[s])
        elif "single" in s:
            single.append(scores[s])
        elif "complete" in s:
            complete.append(scores[s])
        elif "average" in s:
            average.append(scores[s])
    num_clusters = [*range(2,14)]
    # plt.plot(num_clusters, ward, label="ward")
    plt.plot(num_clusters,single, label="single")
    plt.plot(num_clusters,complete, label="complete")
    plt.plot(num_clusters,average, label="average")  
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Different Linkage Methods with Cosine Hierarchical Clustering")  
    plt.legend()
    plt.savefig('parameter_tuning_cos')


graph = loadData('graph_object.pk1')
graph.calcCorellationVectors()
model = TimeSeriesKMeans.from_json('k_means_dtw.json')
model_6 = TimeSeriesKMeans.from_json('k_means_dtw_6.json')
columns = list(graph.datasets.columns)
labels_kmeans = {'epi': 3, 'djia': 1, 'nasdaq': 1, 'russel2000': 1, 's&p500': 1, 'tBond': 1, 'tBill': 1, 'cBond': 1, 'mBond': 1, 'aluminum': 2, 'copper': 2, 'gold': 2, 'lead': 5, 'nickel': 2, 'silver': 7, 'zinc': 1, 'crude': 7, 'natgas': 0, 'petroleum': 7, 'biofuel': 7, 'cattle': 6, 'wheat': 2, 'cocoa': 4, 'coffee': 0, 'corn': 7, 'cotton': 2, 'leanhogs': 6, 'soybeans': 7, 'sugar': 5}
dm = np.load('dtw_matrix.npy')
X = graph.datasets.to_numpy().T

model_agg = AgglomerativeClustering(n_clusters=12, affinity='euclidean' , linkage='ward')
print('fitting model...')
model_agg.fit(X)
print('model fitted')
plotClustersAg(model_agg, 'ward_linkage')


#scores = {'ward_2': 0.4828028341219889, 'single_2': 0.2396248620896954, 'complete_2': 0.19321778732453684, 'average_2': 0.19321778732453684, 'ward_3': 0.4174115078948251, 'single_3': 0.1694336080147411, 'complete_3': 0.32586805681820646, 'average_3': 0.13049829399147367, 'ward_4': 0.3545710702085392, 'single_4': -0.003791573966669423, 'complete_4': 0.2913147797784925, 'average_4': 0.3406174017550323, 'ward_5': 0.24333165791427214, 'single_5': -0.0026101488073381734, 'complete_5': 0.24184523655970924, 'average_5': 0.29959804659001144, 'ward_6': 0.23930092957997834, 'single_6': -0.20062301091298457, 'complete_6': 0.168606233607451, 'average_6': 0.2247145710576746, 'ward_7': 0.23055642668099718, 'single_7': -0.18339388778550736, 'complete_7': 0.15328149479765504, 'average_7': 0.21008758526049892, 'ward_8': 0.2030384188993141, 'single_8': -0.1588180875856671, 'complete_8': 0.1901348234057039, 'average_8': 0.15186314594952402, 'ward_9': 0.22673524423087044, 'single_9': 0.06971253009056788, 'complete_9': 0.1783256010701739, 'average_9': 0.16473419440851883, 'ward_10': 0.19661059894053012, 'single_10': 0.08435286970725112, 'complete_10': 0.17868446936461824, 'average_10': 0.16137377251650342, 'ward_11': 0.20934306937939234, 'single_11': 0.11184505809675703, 'complete_11': 0.1946163765704604, 'average_11': 0.1006342937211491, 'ward_12': 0.24719179770529565, 'single_12': 0.1296040960771252, 'complete_12': 0.15678806881706367, 'average_12': 0.08730762413930862, 'ward_13': 0.14149353931911995, 'single_13': 0.18049572045136794, 'complete_13': 0.17889129877681942, 'average_13': 0.10450303228036781}
# X = to_time_series_dataset(X)




# params = {"metric": "dtw","n_clusters":8,"n_jobs": -1} # {'metric': 'dtw'}


# distance_matrix_correlation = graph.corr_vec
# distance_matrix_dtw = getDtwMatrix(X)
# distance_matrix_dtw = dtw.distance_matrix_fast(X, compact=True)
#dm = getDtwMatrix(X)
# corr = graph.corr_vec.to_numpy()
# print(corr.shape)

# # hierarchicalClustering(corr,columns)
# scores = {}
# for num_cluster in range(2,14):
#     print(num_cluster)
#     for linkage in ['single', 'complete', 'average']:
#         print("..."+linkage)
#         cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='cosine', linkage=linkage) 
#         y_pred = cluster.fit_predict(X) 

#         s = silhouette_score(X, y_pred)
#         print(f'...{s}')
#         scores[f'{linkage}_{num_cluster}'] = s


# scores = {'single_2': 0.2396248620896954, 'complete_2': 0.19321778732453684, 'average_2': 0.19321778732453684, 'single_3': 0.1694336080147411, 'complete_3': 0.32586805681820646, 'average_3': 0.13049829399147367, 'single_4': -0.003791573966669423, 'complete_4': 0.2913147797784925, 'average_4': 0.3406174017550323, 'single_5': -0.0026101488073381734, 'complete_5': 0.24184523655970924, 'average_5': 0.29959804659001144, 'single_6': -0.20062301091298457, 'complete_6': 0.168606233607451, 'average_6': 0.2247145710576746, 'single_7': -0.18339388778550736, 'complete_7': 0.15328149479765504, 'average_7': 0.21008758526049892, 'single_8': -0.1588180875856671, 'complete_8': 0.1901348234057039, 'average_8': 0.15186314594952402, 'single_9': 0.06971253009056788, 'complete_9': 0.1783256010701739, 'average_9': 0.16473419440851883, 'single_10': 0.08435286970725112, 'complete_10': 0.17868446936461824, 'average_10': 0.16137377251650342, 'single_11': 0.11184505809675703, 'complete_11': 0.1946163765704604, 'average_11': 0.1006342937211491, 'single_12': 0.1296040960771252, 'complete_12': 0.15678806881706367, 'average_12': 0.08730762413930862, 'single_13': 0.18049572045136794, 'complete_13': 0.17889129877681942, 'average_13': 0.10450303228036781}
# plotParams(scores)









    




