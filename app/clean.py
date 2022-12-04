import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import List
import datetime
#Python 3.9.5 installed 
#installed numpy, pandas, matplotlib

class DataSet:
    def __init__(self,filename: str, start: str, end: str,name:str):
        self.std_data, self.min_max_data, self.data= self.cleanStockData(filename,start,end,name) 
        
    def cleanStockData(self, file_name : str, start_date_str: str, end_date_str: str, name: str) -> pd.DataFrame:
        #Function Delivers a data frame of soley closing data, normalized
        
        ###WARNING: FILE PATHS ARE RELATIONAL TO THIS FOLDER, DO NOT CHANGE FUNCTION LOCATION BEFORE UPDATING FILE PATHS
            
        #Steps: 
        #1. import csv file
        #2. rename specified columns to specified names
        #3. strip unecessary data
        #4. convert dates to datetime format 
        #5. fill in dates that are not used
        #6. crop data to right length
        #7. normalize
        #8. return 
        
        date_index = 0
        values_index = 1
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        
        #  Import CSV FILE 
        stock_data = pd.read_csv(file_name)
        
        # Rename Specified columns
        ## we will assume "close", "Close" are our main column tags  for actual data, and "date", "Date" as our dates.
        ## if both are not filled we assume that index 0 is date and index 1 is close  (used for edge cases or manual adjustment)
        ##Assumptions: only one date and one close per data set
        
        
        for col_name in stock_data.columns.values:
            if col_name == "Date" or col_name ==  "date":
                #get index of date column and make sure column is properly "Date"
                date_index = stock_data.columns.get_loc(col_name)
                stock_data = stock_data.rename(columns={col_name:"Date"})
            elif col_name == "Close" or col_name == "close":
                #get index of date column and make sure column is properly "Date"
                values_index = stock_data.columns.get_loc(col_name)
                stock_data = stock_data.rename(columns={col_name: name})

        ##check if  our indexes have changed
        
        if date_index == 0:
            stock_data = stock_data.rename(columns={stock_data.columns[0]:"Date"})
        if values_index == 1:
            stock_data = stock_data.rename(columns={stock_data.columns[1]:name})
        
        #Strip Unecessary Data
        
        stock_data =  stock_data.filter(["Date",name])
        
        # Convert Date Column To Datetime Object
        
        stock_data["Date"] = stock_data["Date"].apply(pd.to_datetime)
        
        #Fill in Missing Dates
        ##first define new index including all days
        ## convert dataframe index to date
        ## reindex stock_data using padding method (fills missing values with earliest valid value due to data going from oldest to newest)
        
        new_index = pd.date_range(stock_data["Date"].min(),stock_data["Date"].max())
        stock_data = stock_data.set_index("Date")
        stock_data = stock_data.reindex(new_index,method='pad')
        
        #Crop To Perfered Length
        stock_data = stock_data[start_date:end_date]
        
        #Normalize data:
        std_norm_values = (stock_data[name]-stock_data[name].mean())/stock_data[name].std()    
        min_max_norm_values = (stock_data[name]-stock_data[name].min())/(stock_data[name].max()-stock_data[name].min())
        
        
        
        return std_norm_values, min_max_norm_values, stock_data
        