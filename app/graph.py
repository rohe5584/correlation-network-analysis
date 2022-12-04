import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import List
import datetime
from functools import reduce
from clean import *

class Graph:
    def __init__(self, filename: str , start: str, end: str):
        #read dictionary
        self.start = start
        self.end = end
        self.datasets = []
        self.filename = filename
        self.corr_vec = pd.DataFrame
    
    def buildDataSets(self):
        file_list  = pd.read_csv(self.filename)
        for i in range(len(file_list)):
            self.datasets.append(DataSet(file_list.iloc[i,0],self.start,self.end,file_list.iloc[i,1]).data)
        
    def mergeLists(self)->pd.DataFrame:
        data_merge = reduce(lambda x,y: pd.merge(x,y,left_index=True,right_index=True),self.datasets) 
        return data_merge


    def calcCorellationVectors(self):
        self.datasets = self.mergeLists()
        self.corr_vec = self.datasets.corr()