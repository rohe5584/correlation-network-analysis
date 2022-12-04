import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import List
import datetime
from clean.cleanStockData import *
STOCKPATH = "stock_data/"
BONDPATH = "bond_data/"
COMPATH = "commodity_data/"

## DataSet Class
class DataSet:
    def __init__(self,filename: str, start: str, end: str,type:str):
        
        [self.std_norm, self.min_max_norm, self.stand] = cleanStockData(type+filename,start,end) 
        



russel = DataSet("russel2000_all.csv","2012-12-01","2022-09-01",STOCKPATH)
djia = DataSet("djia_2012.csv","2012-12-01","2022-09-01",STOCKPATH)
nas = DataSet("nasdaq_all.csv","2012-12-01","2022-09-01",STOCKPATH)
gov_corp = DataSet("govt&corp_2012.csv","2012-12-01","2022-09-01",BONDPATH)
tbond = DataSet("tbonds_2012.csv","2012-12-01","2022-09-01",BONDPATH)



#fig, axs = plt.subplots(1,2)
#axs[0].plot(russel.min_max_norm.index,russel.min_max_norm)
#axs[0].plot(djia.min_max_norm.index,djia.min_max_norm)
#axs[0].plot(nas.min_max_norm.index,nas.min_max_norm)



#axs[1].plot(russel.std_norm.index,russel.std_norm)
#axs[1].plot(djia.std_norm.index,djia.std_norm)
#axs[1].plot(nas.std_norm.index,nas.std_norm)
#plt.show()


