import numpy as np
import pandas as pd
from scipy.io import loadmat
import tensorflow.keras as keras

class Datasets:
    BASE = "/media/sda1/dataset/NAB/data"
    DATA = "realAdExchange/"
    CSV1 = "exchange-2_cpc_results.csv"

    MFPT_BASE = "/media/sda1/dataset/MFPT/MFPT Fault Data Sets"
    MFPT_DATA = "1 - Three Baseline Conditions"
    MFPT_MAT = "baseline_1.mat"

class Preprocessing:

    @staticmethod
    def create_dataset(data, window_size):
        
        dataset = np.empty((int(data.size/window_size), window_size, 1))
        j = 0
        for i in range(int(data.size/window_size)):
            dataset[j] = data[i+window_size]
            j = j + 1

        print(f"Dataset shape changed {data.shape} -> {dataset.shape}.")
        print(f"Truncated {data.size - dataset.size} values ")
        
        return dataset

class Model:
    def __init__(self, x_train):
        self.x_train = x_train
    
    def lstm_autoencoder(self, input_size):
        output_size = input_size
        model = keras.models.Sequential([
            keras.layers.RNN(keras.layers.LSTMCell(128), input_shape=[input_size, 1]),
            keras.layers.RepeatVector(output_size),
            keras.layers.RNN(keras.layers.LSTMCell(128), return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(1))
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.x_train, self.x_train, epochs=5)
        result = model.predict(self.x_train)
        return result

if __name__ == '__main__':
    import os
    path = os.path.join(Datasets.MFPT_BASE, Datasets.MFPT_DATA, Datasets.MFPT_MAT)
    
    mat = loadmat(path)
    keys = mat['bearing'].dtype.fields
    mat_dict = {
        list(keys.keys())[0] : mat['bearing'][0][0][0],
        list(keys.keys())[1] : mat['bearing'][0][0][1],
        list(keys.keys())[2] : mat['bearing'][0][0][2],
        list(keys.keys())[3] : mat['bearing'][0][0][3]
    }
    dataset = Preprocessing.create_dataset(mat_dict['gs'], 50)
    
    #model = Model(dataset)
    #result = model.lstm_autoencoder(50)
    #print(result.shape)
    #np.save('mfpt_01.npy', result)
