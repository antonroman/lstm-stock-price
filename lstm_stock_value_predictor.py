#!/usr/bin/env python3
# ./lstm_stock_value_predictor -t AAPL.csv -i AAPL_apr_2019.csv

import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  

from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout  


def main(argv):
    training_samples_file = ''
    testing_samples_file = ''
    stock_name = ''
    try:
        opts, args = getopt.getopt(argv,"ht:i:n:",["trainingfile=","testingfile=","stockname="])
    except getopt.GetoptError:
        print('lstm_stock_value_predictor.py -t <trainingfile> -i <testingfile> -n <stock name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('lstm_stock_value_predictor.py -t <trainingfile> -i <testingfile> -n <stock name>')
            sys.exit()
        elif opt in ("-t", "--training"):
            training_samples_file = arg
        elif opt in ("-i", "--testing"):
            testing_samples_file = arg
        elif opt in ("-n", "--name"):
            stock_name = arg
    print('Training file is: ', training_samples_file)
    print('Testing file is: ', testing_samples_file)
    print('Stock name is: ', stock_name )
    return training_samples_file, testing_samples_file, stock_name

if __name__ == "__main__":
    training_samples_file, testing_samples_file, stock_name = main(sys.argv[1:])
    print(stock_name)

    # reading data from file and get only the first two columns
    #apple_training_complete = pd.read_csv(r'AAPL.csv') 
    apple_training_complete = pd.read_csv(training_samples_file)
    apple_training_processed = apple_training_complete.iloc[:, 1:2].values  

    print(apple_training_processed)
    print(type(apple_training_processed))
    # print(np.isinf(apple_training_processed).any())
    # print(np.isnan(apple_training_processed).any())
    # print(np.argwhere(np.isnan(apple_training_processed)))

    pd.DataFrame(apple_training_processed).to_csv("file_debug.csv")


    # Data normalization
    scaler = MinMaxScaler(feature_range = (0, 1))
    apple_training_scaled = scaler.fit_transform(apple_training_processed)  

    sample_len = len(apple_training_processed)
    print('Trainign sample length: ', len(apple_training_processed))
    features_set = []  
    labels = []  
    N_time_setps = 60

    # we have to predict a value at time T, based on the data from days T-N where N can be any number of steps
    # 60 seems to be an optimal value for optimization, so we need to predict the value at day 61st
    # we need to do sets of 60 values and the lable will be the value at day 61st

    for i in range(N_time_setps, sample_len):  
        features_set.append(apple_training_scaled[i-N_time_setps:i, 0])
        labels.append(apple_training_scaled[i, 0])

    # convert lists into numpy arrays
    features_set, labels = np.array(features_set), np.array(labels)  

    # data must to be shaped to be accepted by LSTM
    # 1st dimension number of records in the dataset, 2n dimension number of time steps
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  

    # instance sequential class
    model = Sequential()  

    # Creating LSTM and Dropout Layers by adding them to the model
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  

    model.add(Dropout(0.2))  

    # add more layers
    model.add(LSTM(units=50, return_sequences=True))  
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))  
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))  
    model.add(Dropout(0.2))  

    # Create dense layer
    model.add(Dense(units = 1))  

    # model compilation, using mean squared error for optimization
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

    # algortihm training
    model.fit(features_set, labels, epochs = 100, batch_size = 32)  

    # test the model with april 2019

    # reading data from file and get only the first two columns
    apple_testing_complete = pd.read_csv(testing_samples_file) 
    apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values  

    apple_samples_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)  


    test_inputs = apple_samples_total[len(apple_samples_total) - len(apple_testing_complete) - 60:].values  

    test_inputs = test_inputs.reshape(-1,1)  
    test_inputs = scaler.transform(test_inputs)

    test_features = []  
    for i in range(60, 80):  
        test_features.append(test_inputs[i-60:i, 0])

    test_features = np.array(test_features)  
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    predictions = model.predict(test_features)  
    predictions = scaler.inverse_transform(predictions)  

    plt.figure(figsize=(10,6))  
    plt.plot(apple_testing_processed, color='blue', label='Actual ' + stock_name + ' Stock Price')  
    plt.plot(predictions , color='red', label='Predicted ' + stock_name + ' Stock Price')  
    plt.title(stock_name + ' Stock Price Prediction')  
    plt.xlabel('Date')  
    plt.ylabel(stock_name + ' Stock Price')  
    plt.legend()  
    plt.show()  

