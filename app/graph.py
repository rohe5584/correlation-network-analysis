import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import datetime
from functools import reduce
from clean import *

class Graph:
    def __init__(self, filename: str , start: str, end: str):
        ## initialize variables
        self.start = start
        self.end = end
        self.datasets = []
        self.filename = filename
        self.corr_vec = pd.DataFrame
        self.G = nx.Graph
    def buildDataSets(self):
        ## create a list from given test data file 
        ##traverse and create DataSet object, and extract data 
        ## kinda convoluted but I expect us to use different data for different reasons so we can add more functions to the data set class when needed
        
        file_list  = pd.read_csv(self.filename)
        for i in range(len(file_list)):
            self.datasets.append(DataSet(file_list.iloc[i,0],self.start,self.end,file_list.iloc[i,1]).std_data)
        
    def mergeLists(self)->pd.DataFrame:
        ## merge list of dataframe data into one dataframe
        data_merge = reduce(lambda x,y: pd.merge(x,y,left_index=True,right_index=True),self.datasets) 
        return data_merge


    def calcCorellationVectors(self):
        ## finaly calculate correlation vector
        self.datasets = self.mergeLists()
        self.corr_vec = self.datasets.corr()
    
    def createNetwork(self)->nx.Graph:
        correlation_matrix = self.corr_vec
        nodes = correlation_matrix.index.values
        correlation_matrix = np.asmatrix(correlation_matrix)

        self.G = nx.from_numpy_matrix(correlation_matrix)
        self.G = nx.relabel_nodes(self.G, lambda i: nodes[i])
        self.G.edges(data=True)
        for e in self.G.edges():
            if(e[0] == e[1]):
                self.G.remove_edge(e[0], e[1])
        return self.G

    def visualizeNetwork(self, G, correlation_direction):
        # fig,ax = plt.subplots(1, figsize=(15,15))

        #VISUALIZE THE WHOLE NETWWORK IN SPRING LAYOUT (will seperate a "root" node)
        plt.figure(figsize=(8,8))
        network = G

        ##caluclate node sizes -> we want nodes with higher degrees to be bigger
        #node_degrees = nx.degree(G)
        degrees = [val for (node, val) in network.degree()]
        node_sizes = [d**3 for d in degrees]
   
        edge, weights = zip(*nx.get_edge_attributes(network, 'weight').items())
        positions = nx.spring_layout(network, k=3.0)
        nx.draw_networkx_nodes(network, positions, node_color='#DA70D6', node_size=node_sizes, alpha=.8)
        nx.draw_networkx_labels(network, positions, font_size=8, font_family='sans-serif')
        if(correlation_direction == 0):
            edges_color = plt.cm.GnBu
            nx.draw_networkx_edges(network, positions, edgelist=edge, style='solid', width =weights, edge_color=weights, edge_cmap=edges_color, edge_vmin = min(weights), edge_vmax=max(weights))
        elif(correlation_direction == 1):
            edges_color = plt.cm.PuRd
            nx.draw_networkx_edges(network, positions, edgelist=edge, style='solid', width =weights, edge_color=weights, edge_cmap=edges_color, edge_vmin = min(weights), edge_vmax=max(weights))
        else: 
            nx.draw_networkx_edges(network, positions, edgelist=edge, style='solid')

        plt.axis('off')
        plt.savefig("network_vis.png", format="PNG")
        plt.show()
    
    def separateCorrelations(self, network, correlation_direction, min_correlation):
        #we seperate the correlations between positive and negative correlations
        #correlation direction <- either 0 or 1 where (0=positive) and (1=negative)

        #rewire graph for only positive correlations
        #i.e. delete all negative edge correlations
        G_seperated = network.copy()
        for index_1, index_2, weight in network.edges.data('weight'):
            if(correlation_direction == 0):
                if(weight < 0.0 or weight < min_correlation):
                    G_seperated.remove_edge(index_1, index_2)
            if(correlation_direction == 1):
                if(weight >= 0.0 or weight > min_correlation):
                    G_seperated.remove_edge(index_1, index_2)
        return G_seperated

    def graph_weighted_edges(self, network, correlation_direction):
        G_weighted = network.copy()
        for index_1, index_2, weight in network.edges.data('weight'):
            if(correlation_direction == 1):
                new_weight = -1*(1+abs(weight))**2
            elif(correlation_direction == 0):
                new_weight = (1+abs(weight))**2   
            G_weighted.remove_edge(index_1, index_2)
            G_weighted.add_edge(index_1, index_2, weight=new_weight)
        return G_weighted