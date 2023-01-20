import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_data(data):
    # Min Max scaler, scales in the range 0-1
    data -= data.min()
    if (data.max() !+)
    data /= data.max()
    return data


if __name__ =="__main__":
    # Reading in the MNIST dataset
    data = pd.read_csv("train.csv")

    # Shuffle the data to be in random order
    data = data.sample(frac=1)

    # Define a test size percentage and split our data for train and testing
    X_data, y_data = data.drop("label", axis=1), data["label"]

    test_size = 0.3 # 30% for test
    test_index = int((1-test_size) * data.shape[0])
    X_train, X_test = X_data.iloc[:test_index], X_data.iloc[test_index:]
    y_train, y_test = y_data.iloc[:test_index], y_data.iloc[test_index:]
    
    # Normalize the data
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    print(X_train.head())
