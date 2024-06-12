import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

from ydata_profiling import ProfileReport

from Data.generate_data import WINDOW_SIZE, measure_memory_usage, generate_X_and_Y

@measure_memory_usage
def load_data(randomize=False):
    # Load X.npy and Y.npy
    X = np.load("Data/data_with_noise/X.npy")
    Y = np.load("Data/data_with_noise/Y.npy")

    # Separate in train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) if not randomize else train_test_split(X, Y, test_size=0.2)

    return X_train, X_test, Y_train, Y_test

def scale_data(X_train, X_test, Y_train, Y_test):
    # Scale The Data
    X_scaler = MinMaxScaler()

    # Reshape the data to 2D
    original_shape_X_train = X_train.shape  
    original_shape_X_test = X_test.shape
    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])

    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Reshape the data back to 3D
    X_train_scaled = X_train_scaled.reshape(original_shape_X_train)
    X_test_scaled = X_test_scaled.reshape(original_shape_X_test)

    Y_scaler = MinMaxScaler()
    Y_train_scaled = Y_scaler.fit_transform(Y_train)
    Y_test_scaled = Y_scaler.transform(Y_test)

    return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler

def descale_data(Y_pred, Y_scaler):
    # Descale the data
    Y_pred_descaled = Y_scaler.inverse_transform(Y_pred)

    return Y_pred_descaled

""" def transform_to_df(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled):
    # Transform numpy to dataframe
    columns = ["idle", "rx", "tx"]
    X_train_df = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=columns)
    Y_train_df = pd.DataFrame(Y_train_scaled, columns=columns)
    Y_test_df = pd.DataFrame(Y_test_scaled, columns=columns)

    return X_train_df, X_test_df, Y_train_df, Y_test_df """

def generate_reports(X_train_df, X_test_df, Y_train_df, Y_test_df):
    # Generate reports
    for name, df in zip(["X_train", "X_test", "Y_train", "Y_test"], [X_train_df, X_test_df, Y_train_df, Y_test_df]):
        profile = ProfileReport(df, title=f"{name} Profiling Report", explorative=True)
        profile.to_file(f"EDAReports/{name}_profiling_report.html")

def preprocess(randomize=False):
    # Load data
    X_train, X_test, Y_train, Y_test = load_data(randomize)

    # Scale data
    X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = scale_data(X_train, X_test, Y_train, Y_test)

    # Transform to dataframe
    # X_train_df, X_test_df, Y_train_df, Y_test_df = transform_to_df(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled)
    #return X_train_df, X_test_df, Y_train_df, Y_test_df, Y_scaler

    return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler


# Test the module
if __name__ == "__main__":

    # Generate the data
    generate_X_and_Y(6)

    # Preprocess the data
    X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = preprocess()
    
    # Generate reports: ONLY WORKS FOR 1 PAST WINDOW and 2D DF
    # generate_reports(X_train_df, X_test_df, Y_train_df, Y_test_df)

    # Print the shape of the data
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"Y_train_scaled shape: {Y_train_scaled.shape}")
    print(f"Y_test_scaled shape: {Y_test_scaled.shape}")
    
