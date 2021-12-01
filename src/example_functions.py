import numpy as np
from sklearn.preprocessing import MinMaxScaler

## Fetch and Store the data
#
# Function Description  : Fetches and Stores the required data into an array
#
# Output Parameters     : data(The Data used for training and prediction)       : data
##
def fetch_store ():
    length_data = 10000
    data = []
    data = [iter for iter in range(length_data)]
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
    # Scaling the data to the range [0, 1]
    scaling = MinMaxScaler(feature_range=(0,1))
    processed_data = scaling.fit_transform(np.array(data).reshape(-1,1))
    
    # Gets the length of the data
    data_length = len(data)

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
    print (inp_arr.shape)
    #print (dependence_interval)
    processed_inp_data =inp_arr.reshape(inp_arr.shape[0],inp_arr.shape[1] , 1)
    print(processed_inp_data.shape)
    
    return processed_inp_data, np.array(exp_data)