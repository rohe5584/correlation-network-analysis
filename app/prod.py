import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import datetime
from clean import *
from graph import *

import sys

graph = Graph(sys.argv[1],"2012-12-01","2022-09-01")
graph.buildDataSets()
graph.calcCorellationVectors()
print(graph.corr_vec)
graph.createNetwork()
print(graph.G)
#graph.visualizeNetwork(graph.G)
print(type(graph.corr_vec))
print(type(graph.G))
positive_corr_network = graph.separateCorrelations(graph.G, 0, 0.7)
negative_corr_network = graph.separateCorrelations(graph.G, 1, -0.1)
print(positive_corr_network)
print(negative_corr_network)
positive_weighted_corr_network = graph.graph_weighted_edges(positive_corr_network, 0)
negative_weighted_corr_network = graph.graph_weighted_edges(negative_corr_network, 1)
graph.visualizeNetwork(negative_weighted_corr_network, 1)
graph.visualizeNetwork(positive_weighted_corr_network, 0)
#graph.visualizeNetwork(negative_corr_network)