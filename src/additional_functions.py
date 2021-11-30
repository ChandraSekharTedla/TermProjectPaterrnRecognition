import numpy as np

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


## Split Data Function
#
# Function Description  : Splits data into 2 sets : Train Data, Test Data
#
# Input Parameters      : 1. data (The data that is to be splitted)                             : data
#                         2. percentage_train_data (The percentage of data used for training)   : percentage_train_data
#
# Output Parameters     : 1. Train Data (1st part of Data)                                      : train_data
#                         2. Test Data (2nd part of Data)                                       : test_data
##
def split_data (data, percentage_train_data):
    # Gets the length of the data
    data_length = len(data)

    # Gets the length of the Training data
    train_data_length = int (percentage_train_data*data_length)

    # Splits the data into 2 sets : Train data and Test data
    train_data = data[:train_data_length]
    test_data = data[train_data_length:]

    return train_data, test_data


## Preprocessing of Data
#
# Function Description  : Precprocesses the train and test data as per the crierion of the Neural Networks.
#
# Criterion             : Scale data to range [0, 1]
#
# Input Parameters      : 1. Train Data                 : train_data
#                         2. Test Data                  : test_data
#
# Output Parameters     : 1. Processed Train Data       : processed_train_data
#                         2. Processed Test Data        : processed_test_data
#                         3. Scaling Factor used        : scaling_factor
##
def preprocessing_data (train_data, test_data):
    # Finding maximum among the Train and Test Data
    max_train_data = max(train_data)
    max_test_data = max(test_data)

    scaling_factor = max(max_train_data, max_test_data)

    print (scaling_factor)    

    # Scaling the data to the range [0, 1]
    processed_train_data = float(train_data/scaling_factor)
    processed_test_data = float(test_data/scaling_factor)

    return processed_train_data, processed_test_data, scaling_factor


## Reverting the Scaling
#
# Function Description  : Removes the scaling used during the preprocessing
#
# Input Parameters      : 1. data (Data to be scaled)   : data
#                         2. Scaling Factor             : scaling_factor
#
# Output Parameters     : 1. Processed data             : scaled_data
##
def invert_scaling (data, scaling_factor):
    # Remove the scaling
    scaled_data = data*scaling_factor

    return scaled_data