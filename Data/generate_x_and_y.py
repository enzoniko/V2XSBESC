import numpy as np
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt

WINDOW_SIZE = 5400 # in miliseconds
DATA_PATH = 'Data/iot/' # Leave the last slash

# Sturges helper function
def sturges(n: int):
    result = np.ceil(np.log2(n))
    return int(result) + 1


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

    num_bins = sturges(len(all_tx_durations))
    axs[0].hist(all_tx_durations, bins=num_bins, color='blue', log=True)
    axs[0].set_title('TX Durations', fontsize=18)
    axs[0].set_xlabel('Duration (ms)', fontsize=18)
    axs[0].set_ylabel('Frequency', fontsize=18)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].tick_params(axis='x', labelsize=18)
    axs[0].tick_params(axis='y', labelsize=18)

    mean_tx = np.mean(all_tx_durations)
    median_tx = np.median(all_tx_durations)
    axs[0].axvline(mean_tx, color='black', linestyle='--', linewidth=2)
    axs[0].axvline(median_tx, color='black', linestyle='solid', linewidth=2)
   

    num_bins = sturges(len(all_rx_durations))
    axs[1].hist(all_rx_durations, bins=num_bins, color='green', log=True)
    axs[1].set_title('RX Durations', fontsize=18)
    axs[1].set_xlabel('Duration (ms)', fontsize=18)
    axs[1].set_ylabel('Frequency', fontsize=18)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].tick_params(axis='x', labelsize=18)
    axs[1].tick_params(axis='y', labelsize=18)

    mean_rx = np.mean(all_rx_durations)
    median_rx = np.median(all_rx_durations)
    axs[1].axvline(mean_rx, color='black', linestyle='--', linewidth=2)
    axs[1].axvline(median_rx, color='black', linestyle='solid', linewidth=2)
  

    num_bins = sturges(len(all_idle_durations))
    axs[2].hist(all_idle_durations, bins=num_bins, color='red', log=True)
    axs[2].set_title('Idle Durations', fontsize=18)
    axs[2].set_xlabel('Duration (ms)', fontsize=18)
    axs[2].set_ylabel('Frequency', fontsize=18)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].tick_params(axis='x', labelsize=18)
    axs[2].tick_params(axis='y', labelsize=18)

    mean_idle = np.mean(all_idle_durations)
    median_idle = np.median(all_idle_durations)
    axs[2].axvline(mean_idle, color='black', linestyle='--', linewidth=2, label=f'Mean')
    axs[2].axvline(median_idle, color='black', linestyle='solid', linewidth=2, label=f'Median')

    # Add legend outside the subplots
    fig.legend(loc='upper right', fontsize=18)

    plt.tight_layout()


    plt.savefig(f'{DATA_PATH}summarized_windowed_durations_histogram.pdf')

def read_vehicle_data(vehicle_key):
    print('.', end='', flush=True)
    filename = f'{DATA_PATH}windows_summarized_{WINDOW_SIZE}ms.hdf5'
    with h5py.File(filename, 'r') as file:
        vehicle_data = []
        for window_key in file[vehicle_key].keys():
            window_data = file[f'{vehicle_key}/{window_key}'][:]
            window_data = np.char.decode(window_data, 'utf-8').astype(float)
            vehicle_data.append(window_data)
        return vehicle_key, np.array(vehicle_data)

def generate_X_and_Y(num_past_windows: int = 1):
    print("Generating X and Y data")

    filename = f'{DATA_PATH}windows_summarized_{WINDOW_SIZE}ms.hdf5'
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
        
        for i in range(0, len(vehicle_data) - num_past_windows - 1, 1): # changed step to 1 from num_past_windows + 1
            if i + num_past_windows < len(vehicle_data):
                X.append(vehicle_data[i:i+num_past_windows])
                idle, rx, tx = vehicle_data[i+num_past_windows]
                Y.append((idle, rx))

       

    X = np.array(X)
    Y = np.array(Y)
    from scipy.stats import skew, kurtosis
    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    X_scaler = MinMaxScaler()
    X_ = X_scaler.fit_transform(X.reshape(-1, 3)).reshape(-1, num_past_windows, 3)
    Y_scaler = MinMaxScaler()
    Y_ = Y_scaler.fit_transform(Y)

    # Calculate the error of using the mean of the past windows as the prediction
    maes_per_vehicle = []
    for i in range(len(X_) - 1):
        mean_idle = np.mean(X_[i, :, 0])
        mean_rx = np.mean(X_[i, :, 1])
        mae_idle = np.abs(mean_idle - Y_[i, 0])
        mae_rx = np.abs(mean_rx - Y_[i, 1])

        mean_tx = np.mean(X_[i, :, 2])
        tx_true = X_[i+1, 0, 2]
        mae_tx = np.abs(mean_tx - tx_true)

        maes_per_vehicle.append((mae_idle, mae_rx, mae_tx))
    
    maes_per_vehicle = np.array(maes_per_vehicle)
    print(f"Mean MAE Idle: {np.mean(maes_per_vehicle[:, 0])}")
    print(f"Mean MAE RX: {np.mean(maes_per_vehicle[:, 1])}")
    print(f"Mean MAE TX: {np.mean(maes_per_vehicle[:, 2])}")
    print(f"STD MAE Idle: {np.std(maes_per_vehicle[:, 0])}")
    print(f"STD MAE RX: {np.std(maes_per_vehicle[:, 1])}")
    print(f"STD MAE TX: {np.std(maes_per_vehicle[:, 2])}")
    overall_mae = np.mean(maes_per_vehicle[:, :2])
    print(f"Overall MAE: {overall_mae}")



    plot_summarized_durations(all_data)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    np.save(f'{DATA_PATH}X.npy', X)
    np.save(f'{DATA_PATH}Y.npy', Y)

    # Calculate Mean, Median, Std, Skewness, and Kurtosis of X for rx, tx, and idle
    print("Calculating statistics of X")
    X = X.reshape(-1, 3)
    mean_idle = np.mean(X[:, 0])
    mean_rx = np.mean(X[:, 1])
    mean_tx = np.mean(X[:, 2])
    median_idle = np.median(X[:, 0])
    median_rx = np.median(X[:, 1])
    median_tx = np.median(X[:, 2])
    std_idle = np.std(X[:, 0])
    std_rx = np.std(X[:, 1])
    std_tx = np.std(X[:, 2])
    skew_idle = skew(X[:, 0])
    skew_rx = skew(X[:, 1])
    skew_tx = skew(X[:, 2])
    kurt_idle = kurtosis(X[:, 0])
    kurt_rx = kurtosis(X[:, 1])
    kurt_tx = kurtosis(X[:, 2])
    print(f"Mean Idle: {mean_idle}")
    print(f"Mean RX: {mean_rx}")
    print(f"Mean TX: {mean_tx}")
    print(f"Median Idle: {median_idle}")
    print(f"Median RX: {median_rx}")
    print(f"Median TX: {median_tx}")
    print(f"Std Idle: {std_idle}")
    print(f"Std RX: {std_rx}")
    print(f"Std TX: {std_tx}")
    print(f"Skew Idle: {skew_idle}")
    print(f"Skew RX: {skew_rx}")
    print(f"Skew TX: {skew_tx}")
    print(f"Kurt Idle: {kurt_idle}")
    print(f"Kurt RX: {kurt_rx}")
    print(f"Kurt TX: {kurt_tx}")

    return overall_mae



if __name__ == "__main__":

    overall_mae = generate_X_and_Y(num_past_windows=2)