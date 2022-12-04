import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import List
import datetime
from clean import *
from graph import *

import sys

graph = Graph(sys.argv[1],"2012-12-01","2022-09-01")
graph.buildDataSets()
graph.calcCorellationVectors()
print(graph.corr_vec)




