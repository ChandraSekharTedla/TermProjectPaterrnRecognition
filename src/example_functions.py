import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

## Fetch and Store the data
#
# Function Description  : Fetches and Stores the required data into an array
#
# Output Parameters     : data(The Data used for training and prediction)       : data
##
def fetch_store ():
    basic = 0
    if (basic == 1):
        length_data = 10000
        data = []
        data = [iter for iter in range(length_data)]
        
    elif(basic == 0):
        #Store data from the file
        data_frame = pd.read_csv("TSLA.csv") 
        
        #name_of_dat_frame.name_of_column
        data = data_frame.Close
    return data


## Preprocessing of Data
#
# Function Description  : Precprocesses the train and test data as per the crierion of the Neural Networks.
#
# Criterion             : Scale data to range [0, 1]
#
# Input Parameters      : 1. data (The data that is to be splitted)                             : data
#                         2. percentage_train_data (The percentage of data used for training)   : percentage_train_data
#
# Output Parameters     : 1. Scaling                    : processed_train_data
#                         2. Processed Data             : processed_data
#                         3. Processed Train Data       : processed_train_data
#                         4. Processed Test Data        : processed_test_data
##
def preprocessing_data (data, percentage_train_data):
    # Taking Log Transform for the data
    log_transformed_data = np.log(data)
    
    # Scaling the data to the range [0, 1]
    scaling = MinMaxScaler(feature_range=(0,1))
    processed_data = scaling.fit_transform(np.array(log_transformed_data).reshape(-1,1))
    
    # Gets the length of the data
    data_length = len(log_transformed_data)

    # Gets the length of the Training data
    train_data_length = int (percentage_train_data*data_length)

    # Splits the data into 2 sets : Train data and Test data
    processed_train_data = processed_data[:train_data_length]
    processed_test_data = processed_data[train_data_length:]
    
    return scaling, processed_data, processed_train_data, processed_test_data


## Generate Sequence used by the LSTM
#
# Function Description  : Places the data in such a way that LSTM can train the model
#
# Input Parameters      : 1. data (Data to be scaled)   : data
#                         2. Features used              : dependence_interval
#
# Output Parameters     : Sequence Used by LSTM         : processed_inp_data, exp_data
##
# convert an array of values into a dataset matrix
def generate_sequence(data, dependence_interval):
    inp_data, exp_data = [], []
    
    for iter in range(len(data)-dependence_interval-1):
        temp_array = data[iter:(iter+dependence_interval), 0]
        inp_data.append(temp_array)
        exp_data.append(data[iter + dependence_interval, 0])
        
    inp_arr = np.array(inp_data)
#     print ("inp_arr shape : ", inp_arr.shape)
#     print ("depe_interval : ", dependence_interval)
    processed_inp_data =inp_arr.reshape(inp_arr.shape[0],inp_arr.shape[1] , 1)
#     print ("inp_arr shape : ", processed_inp_data.shape)
    
    return processed_inp_data, np.array(exp_data)


## LSTM Training and Tesing
#
# Function Description  : Trains and tests LSTM Model
#
# Input Parameters      : 1. data (Data to be scaled)                  : data
#                         2. Features used                             : dependence_interval
#                         3. The percentage of data used for training  : percentage_train_data
#                         4. Number of epochs for LSTM                 : num_epochs
#
# Output Parameters     : Sequence Used by LSTM         : processed_inp_data, exp_data
##
# convert an array of values into a dataset matrix
def lstm (data, percentage_train_data, dependence_interval, num_epochs):
    ## Step - 1
    # Preprocesses the data for LSTM and Splits the data among training and testing 
    scaler, processed_data, processed_train_data, processed_test_data = preprocessing_data (data, percentage_train_data)


    ## Step - 2
    # Creating Training and Test Sequence and the corresponding otput sequence from the data
    train_sequence, expected_train_output_seq = generate_sequence(processed_train_data, dependence_interval)
    test_sequence, expected_test_output_seq = generate_sequence(processed_test_data, dependence_interval)

    
    ## Step - 3
    # Call LSTM
    model=Sequential()
    # Check how this LSTM model works 
    model.add(LSTM(50,return_sequences=True,input_shape=(train_sequence.shape[1],train_sequence.shape[2])))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    # Check
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.summary()

    model.fit(train_sequence,expected_train_output_seq,validation_data=(test_sequence,expected_test_output_seq),epochs=num_epochs,batch_size=64,verbose=1)

    
    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(train_sequence)
    test_predict = model.predict(test_sequence)

    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    ##Removing log transform
    train_predict = np.exp(train_predict)
    test_predict = np.exp(test_predict)

    ### Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error
    print("Empirical Error is ", math.sqrt(mean_squared_error(expected_train_output_seq, train_predict)))

    ### Test Data RMSE
    print("True Error is ", math.sqrt(mean_squared_error(expected_test_output_seq,test_predict)))
    
    ### Plotting 
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(processed_data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[dependence_interval:len(train_predict)+dependence_interval, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(processed_data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(dependence_interval*2)+1:len(processed_data)-1, :] = test_predict
    # plot baseline and predictions
    plt.figure(figsize=[15,10])
    plt.plot(data)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    
    return model, scaler, processed_data, train_predict, test_predict