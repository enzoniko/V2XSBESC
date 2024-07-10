import pandas as pd
import numpy as np
import h5py
from itertools import permutations
import matplotlib.pyplot as plt

WINDOW_SIZE_IN_SECONDS = 14
DATA_PATH = 'Data/data_with_noise/'

def load_windowed_data(file_path):
    windowed_data = {}
    num_windows = []
    with h5py.File(file_path, 'r') as file:
        vehicles = list(file.keys())
        total_vehicles = len(vehicles)
        for i, vehicle in enumerate(vehicles):
            vehicle_windows = []
            for window in file[vehicle].keys():
                data = file[f'{vehicle}/{window}'][:]
                vehicle_windows.append(data)
            windowed_data[vehicle] = vehicle_windows
            num_windows.append(len(vehicle_windows))
            percentage_processed = (i + 1) / total_vehicles * 100
            print(f"Percentage of vehicles processed: {percentage_processed:.2f}%", flush=True)
    

    # Plot histogram of number of windows
    plt.hist(num_windows, bins=20)
    plt.xlabel("Number of Windows")
    plt.ylabel("Frequency")
    plt.title("Histogram of Number of Windows")
    plt.show()

    # Sort the list of windows per vehicle
    sorted_windows = sorted(num_windows)

    # Calculate the median number of windows
    median_windows = np.percentile(sorted_windows, 50)

    # Find the number of windows required to reduce the amount of vehicles to 50%
    # This is effectively finding the median of the number of windows per vehicle
    num_windows_for_50_percent = int(np.ceil(median_windows))

    print(f"Median number of windows: {median_windows:.2f}")
    print(f"Number of windows required to cover 50% of vehicles: {num_windows_for_50_percent}")

    avg_num_windows = np.mean(num_windows)
    std_num_windows = np.std(num_windows)
    print(f"Average number of windows: {avg_num_windows:.2f}")
    print(f"Standard deviation of number of windows: {std_num_windows:.2f}")
    
    return windowed_data

def count_mode_switches(windowed_data):
    switch_counts = []
    for vehicle, windows in windowed_data.items():
        for window in windows:
            window_data = [entry[0].decode('utf-8') for entry in window]
            window_switch_counts = {}
            for mode1, mode2 in permutations(['rx', 'tx', 'idle'], 2):
                count = sum((window_data[j] == mode1 and window_data[j + 1] == mode2)
                            for j in range(len(window_data) - 1))
                window_switch_counts[(mode1, mode2)] = count
            switch_counts.append(window_switch_counts)
    return switch_counts

def calculate_cv_for_switches(switch_counts):
    switch_durations = {key: [] for key in permutations(['rx', 'tx', 'idle'], 2)}
    
    for switch_count in switch_counts:
        for switch, count in switch_count.items():
            switch_durations[switch].append(count)
    
    cv_results = {}
    for switch, durations in switch_durations.items():
        if len(durations) > 1:
            cv = np.std(durations) / np.mean(durations) if np.mean(durations) != 0 else np.nan
            cv_results[switch] = cv
        else:
            cv_results[switch] = np.nan
    return cv_results, switch_durations

def plot_histogram(switch_durations):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, switch in enumerate(switch_durations.keys()):
        durations = switch_durations[switch]
        ax = axs[i // 3, i % 3]
        ax.hist(durations, bins=20)
        ax.set_title(f"{switch[0]} to {switch[1]}", fontsize=18)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = f'{DATA_PATH}windows{WINDOW_SIZE_IN_SECONDS}s.hdf5'
    windowed_data = load_windowed_data(file_path)
    
    switch_counts = count_mode_switches(windowed_data)
    cv_results, switch_durations = calculate_cv_for_switches(switch_counts)
    
    print("Coefficient of Variation for each mode switch:")
    for switch, cv in cv_results.items():
        print(f"{switch}: {cv:.4f}")


    plot_histogram(switch_durations)