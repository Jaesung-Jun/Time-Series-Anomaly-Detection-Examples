from curses import A_ALTCHARSET
import numpy as np
import tensorflow.keras as keras
from matplotlib import pyplot as plt

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series = 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

class Measure_Baseline:
    @staticmethod
    def simple_linear_model(x_train, y_train, x_val, y_val):
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[50, 1]),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
        model.fit(x=x_train, y=y_train, epochs=20)
        acc = model.evaluate(x_val, y_val)
        return acc
    @staticmethod
    def naive_forecasting(x_val, y_val):
        y_pred = x_val[:, -1]
        return np.mean(keras.losses.mean_squared_error(y_val, y_pred))

class Model:
    def __init__(self, x_train, y_train, x_test, y_test, x_valid, y_valid):
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test

        self.x_val = x_valid
        self.y_val = y_valid
    
    def simple_rnn(self):
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(1, input_shape=[None, 1])
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        model.fit(self.x_train, self.y_train, epochs=20)
        acc = model.evaluate(self.x_val, self.y_val)
        return acc 
    
    def deep_rnn(self):
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        model.fit(self.x_train, self.y_train, epochs=20)
        acc = model.evaluate(self.x_val, self.y_val)
        return acc 

    def __last_time_step_mse(self, y_true, y_pred):
        return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])

    def time_distributed_rnn(self):
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.__last_time_step_mse])
        model.fit(self.x_train, self.y_train, epochs=20)
        acc = model.evaluate(self.x_val, self.y_val)
        return acc[1]

    def time_distributed_lstm(self):
        model = keras.models.Sequential([
            keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True, input_shape=[None, 1]),
            keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.__last_time_step_mse])
        model.fit(self.x_train, self.y_train, epochs=20)
        acc = model.evaluate(self.x_val, self.y_val)
        return acc[1]
    
if __name__ == '__main__':
    n_steps = 50
    ex_series = generate_time_series(10000, n_steps + 1)   # last one is Ground-truth for measuring baseline performance

    x_train, y_train = ex_series[:7000, :n_steps], ex_series[:7000, -1]
    x_valid, y_valid = ex_series[7000:9000, :n_steps], ex_series[7000:9000, -1]
    x_test, y_test = ex_series[9000:, :n_steps], ex_series[9000:, -1]

    results = {}
    print(x_train.shape)
    """
    model = Model(x_train, y_train, x_test, y_test, x_valid, y_valid)

    results['simple_linear_model_baseline'] = Measure_Baseline.simple_linear_model(x_train, y_train, x_valid, y_valid)
    results['naive_forecast_baseline'] = Measure_Baseline.naive_forecasting(x_valid, y_valid)

    #results['simple_rnn'] = model.simple_rnn()
    

    ####################################################################################################################

    ex_series = generate_time_series(10000, n_steps + 10)   
    x_train, y_train = ex_series[:7000, :n_steps], ex_series[:7000, -10:, 0]
    x_valid, y_valid = ex_series[7000:9000, :n_steps], ex_series[7000:9000, -10:, 0]
    x_test, y_test = ex_series[9000:, :n_steps], ex_series[9000:, -10:, 0]

    #model = Model(x_train, y_train, x_test, y_test, x_valid, y_valid)
    #results['deep_rnn'] = model.deep_rnn() 

    ####################################################################################################################
    
    ex_series = generate_time_series(10000, n_steps + 10) 
    y = np.empty((10000, n_steps, 10))
    for step_ahed in range(1, 10+1):
        y[:, :, step_ahed - 1] = ex_series[:, step_ahed:step_ahed + n_steps, 0]
    y_train = y[:7000]
    y_valid = y[7000:9000]
    y_test = y[9000:]
    
    model = Model(x_train, y_train, x_test, y_test, x_valid, y_valid)
    results['time_distributed_rnn'] = model.time_distributed_rnn() 
    results['time_distributed_lstm'] = model.time_distributed_lstm() 
#plt.plot(example_time_series[0])
#plt.savefig('test.png')
"""