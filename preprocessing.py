import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

from ydata_profiling import ProfileReport

from Data.generate_data import WINDOW_SIZE

# Create a decorator to measure the memory usage
def measure_memory_usage(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss  # Resident Set Size (RAM usage)

        # Execute the function
        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss
        print(f"Memory usage: {(mem_after - mem_before)/1024**3} GB")

        return result

    return wrapper

@measure_memory_usage
def load_data():
    # Load X.npy and Y.npy
    X = np.load("Data/X.npy")
    Y = np.load("Data/Y.npy")

    # Separate in train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test

def scale_data(X_train, X_test, Y_train, Y_test):
    # Scale The Data
    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    Y_scaler = MinMaxScaler()
    Y_train_scaled = Y_scaler.fit_transform(Y_train)
    Y_test_scaled = Y_scaler.transform(Y_test)

    return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled

def transform_to_df(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled):
    # Transform numpy to dataframe
    columns = ["idle", "rx", "tx"]
    X_train_df = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=columns)
    Y_train_df = pd.DataFrame(Y_train_scaled, columns=columns)
    Y_test_df = pd.DataFrame(Y_test_scaled, columns=columns)

    return X_train_df, X_test_df, Y_train_df, Y_test_df

def generate_reports(X_train_df, X_test_df, Y_train_df, Y_test_df):
    # Generate reports
    for name, df in zip(["X_train", "X_test", "Y_train", "Y_test"], [X_train_df, X_test_df, Y_train_df, Y_test_df]):
        profile = ProfileReport(df, title=f"{name} Profiling Report", explorative=True)
        profile.to_file(f"EDAReports/{name}_profiling_report.html")

def preprocess():
    # Load data
    X_train, X_test, Y_train, Y_test = load_data()

    # Scale data
    X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = scale_data(X_train, X_test, Y_train, Y_test)

    # Transform to dataframe
    X_train_df, X_test_df, Y_train_df, Y_test_df = transform_to_df(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled)

    return X_train_df, X_test_df, Y_train_df, Y_test_df

# Test the module
if __name__ == "__main__":

    # Preprocess the data
    X_train_df, X_test_df, Y_train_df, Y_test_df = preprocess()
    
    # Generate reports
    generate_reports(X_train_df, X_test_df, Y_train_df, Y_test_df)
