import pandas as pd
class FeatureEngineering:
    def __init__(self, ds):
        self.ds = ds

    def task1(self):
        columns = self.ds.columns
        c_types = self.ds.dtypes
        for i in range(len(columns)):
            if(i!=0):
                if(c_types[i] == float):
                    average = self.ds[columns[i]].mean()
                    self.ds[columns[i]].fillna(average, inplace = True)


    def task2(self):
        columns = self.ds.columns
        c_types = self.ds.dtypes
        for i in range(len(columns)):
            if(i!=0):
                if(c_types[i] == float):
                    self.ds[columns[i]] = (self.ds[columns[i]] - self.ds[columns[i]].mean())/self.ds[columns[i]].std()
        




