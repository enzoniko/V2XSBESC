import numpy as np
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt

WINDOW_SIZE = 14 # in miliseconds
DATA_PATH = 'Data/data_with_noise/' # Leave the last slash

def plot_summarized_durations(all_data):

    all_tx_durations = []

    all_rx_durations = []

    all_idle_durations = []

    # Plot the tx durations for each vehicle
    for vehicle_key, vehicle_data in all_data.items():
   
        for window in vehicle_data:
            all_idle_durations.append(float(window[0]))
            all_rx_durations.append(float(window[1]))
            all_tx_durations.append(float(window[2]))

    # Plot the distribution of durations in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].hist(all_tx_durations, bins=100, color='blue', log=True)
    axs[0].set_title('TX Durations')
    axs[0].set_xlabel('Duration (ms)')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(all_rx_durations, bins=100, color='green', log=True)
    axs[1].set_title('RX Durations')
    axs[1].set_xlabel('Duration (ms)')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(all_idle_durations, bins=100, color='red', log=True)
    axs[2].set_title('Idle Durations')
    axs[2].set_xlabel('Duration (ms)')
    axs[2].set_ylabel('Frequency')

    plt.savefig(f'{DATA_PATH}summarized_windowed_durations_histogram.png')

def read_vehicle_data(vehicle_key):
    print('.', end='', flush=True)
    filename = f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5'
    with h5py.File(filename, 'r') as file:
        vehicle_data = []
        for window_key in file[vehicle_key].keys():
            window_data = file[f'{vehicle_key}/{window_key}'][:]
            window_data = np.char.decode(window_data, 'utf-8').astype(float)
            vehicle_data.append(window_data)
        return vehicle_key, np.array(vehicle_data)

def generate_X_and_Y(num_past_windows: int = 1):
    print("Generating X and Y data")

    filename = f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5'
    with h5py.File(filename, 'r') as file:
        print('-', end='', flush=True)
        vehicle_keys = list(file.keys())

    with Pool() as pool:
        results = pool.map(read_vehicle_data, vehicle_keys)
        
    all_data = {vehicle_key: vehicle_data for vehicle_key, vehicle_data in results}

    # Create the X and Y data
    X, Y = [], []
    for vehicle_key, vehicle_data in all_data.items():
        if len(vehicle_data) < num_past_windows + 1:
            continue

        for i in range(0, len(vehicle_data) - num_past_windows - 1, num_past_windows + 1):
            if i + num_past_windows < len(vehicle_data):
                X.append(vehicle_data[i:i+num_past_windows])
                idle, rx, _ = vehicle_data[i+num_past_windows]
                Y.append((idle, rx))

    X = np.array(X)
    Y = np.array(Y)

    plot_summarized_durations(all_data)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    np.save(f'{DATA_PATH}X.npy', X)
    np.save(f'{DATA_PATH}Y.npy', Y)

if __name__ == "__main__":

    generate_X_and_Y(num_past_windows=2)