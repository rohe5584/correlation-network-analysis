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
    

    
graph = loadData('graph_object.pk1')

X = graph.datasets.to_numpy().T
X = to_time_series_dataset(X)

params = {"metric": "dtw", "n_jobs": 2} # {'metric': 'dtw'}
testModel(params, X, 20, 'K_means_inertia_dtw.png', 'K_means_silhouette_dtw.png')






    




