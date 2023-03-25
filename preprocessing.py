import pandas as pd
import numpy as np
import os
import json

from torch import parse_schema
from osAdvanced import File_Control

class NAB:
    def __init__(self, path, data_name, json_path):
        self.paths_by_dir = File_Control.searchAllFilesInDirectoryByDir(path, 'csv', progressbar=True)
        self.paths = [path for paths in self.paths_by_dir 
                           for path in paths if path.find(data_name) != -1]
        self.csvs = {}
        self.data_name = data_name
        self.json_path = json_path
        
    def load_train_data(self):
        for path in self.paths:
            csv = pd.read_csv(path)
            path = path.replace(path.split(f'{self.data_name}')[0], "")
            self.csvs[path] = csv
        
        anomaly_timestamp = self.load_anomaly_timestamp()
        
        for df in self.csvs:
            df_anomaly = anomaly_timestamp[df]
            flag = False
            is_anomaly = []
            i = 0
            for timestamp in self.csvs[df]['timestamp'].values:
                #print(timestamp, df_anomaly[i][0].split('.')[0])
                if df_anomaly != []:
                    if timestamp == df_anomaly[i][0].split('.')[0]:
                        flag = True
                    if flag == True:
                        is_anomaly.append(True)
                    elif flag == False:
                        is_anomaly.append(False)
                    if timestamp == df_anomaly[i][1].split('.')[0]:
                        flag = False
                        i = i + 1
                        if i == len(df_anomaly):
                            i = len(df_anomaly)-1
                else:
                    is_anomaly.append(False)
            self.csvs[df]['is_anomaly'] = is_anomaly
            self.csvs[df] = self.csvs[df].values
            
            #print(len(is_anomaly), self.csvs[df]['timestamp'].values.size)
        return self.csvs
    def load_anomaly_timestamp(self):
        timestamp_dict = {}
        
        with open(self.json_path, 'r') as file:
            label_json = json.load(file)
        for path in label_json:
            if path.find(self.data_name) != -1:
                timestamp_dict[path] = label_json[path]
                         
        return timestamp_dict
                
class NASA:
    def __init__ (self, path):
        self.path = path
        self.msl_list
        self.smap_list 
    def load_data(self, data_type):
        self.paths_by_dir = File_Control.searchAllFilesInDirectory(os.path.join(self.path, data_type), 'npy', progressbar=True)
        print(self.paths_by_dir)
        for path in self.paths_by_dir:
            test = np.load(path)
            print(path, ":", test.shape)
    
    def load_label(self, path):
        csv = pd.read_csv(path)
        
class CWRU:
    pass

"""
path = '/media/sda1/dataset/NAB'
data_name = 'realAWSCloudwatch'
json_path = os.path.join(path, 'labels', 'combined_windows.json')

nab = NAB(os.path.join(path, 'data', data_name), data_name, json_path)
train_set = nab.load_train_data()
print(train_set)
"""

path = '/media/sda1/dataset/NASA'
nasa = NASA(path)
nasa.load_data('train')
