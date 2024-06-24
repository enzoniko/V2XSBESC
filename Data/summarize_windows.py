import h5py
import os
import dask.array as da
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

WINDOW_SIZE = 14 # in miliseconds
DATA_PATH = 'Data/data_with_noise/' # Leave the last slash


def process_vehicle_chunk(vehicle_keys, input_file_path, output_file_path, lock):
    with h5py.File(input_file_path, 'r') as input_file:
        vehicle_data = {}
        for vehicle_key in vehicle_keys:
            vehicle_data[vehicle_key] = []
            # Iterate through the windows of each vehicle
            for window_key in input_file[vehicle_key].keys():
                # Use Dask to handle large data
                dset = input_file[f'{vehicle_key}/{window_key}']
                
                # Define the chunks size according to the dataset's dimensions
                chunks = tuple(min(1000, dim) for dim in dset.shape)
                
                window_data = da.from_array(dset, chunks=chunks)
                
                # Compute and convert the data if needed
                window_data = np.char.decode(window_data.compute(), 'utf-8')
                
                # Append to the vehicle's data
                vehicle_data[vehicle_key].append(window_data)

            # Process the vehicle's data
            for i in range(len(vehicle_data[vehicle_key])):
                idle = 0
                rx = 0
                tx = 0
                for j in range(len(vehicle_data[vehicle_key][i])):
                    if vehicle_data[vehicle_key][i][j][0] == 'idle':
                        idle += float(vehicle_data[vehicle_key][i][j][1])
                    elif vehicle_data[vehicle_key][i][j][0] == 'rx':
                        rx += float(vehicle_data[vehicle_key][i][j][1])
                    elif vehicle_data[vehicle_key][i][j][0] == 'tx':
                        tx += float(vehicle_data[vehicle_key][i][j][1])
                
                vehicle_data[vehicle_key][i] = [idle, rx, tx]

    # Write the processed data to the output file with a lock
    with lock:
        with h5py.File(output_file_path, 'a') as output_file:
            for vehicle_key, data in vehicle_data.items():
                for j, window in enumerate(data):
                    window = np.array(window, dtype='S32')
                    output_file.create_dataset(f'vehicle_{vehicle_key}/window_{j}', data=window)

def chunkify(lst, n):
    """Divide a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sum_mode_durations_in_windows():
    print("Summarizing mode durations in windows")
    total_number_of_vehicles = 0
    # Remove the previous file if it exists
    summarized_file_path = f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5'
    try:
        os.remove(summarized_file_path)
    except FileNotFoundError:
        pass

    input_file_path = f'{DATA_PATH}windows{WINDOW_SIZE}s.hdf5'
    output_file_path = summarized_file_path

    with h5py.File(input_file_path, 'r') as input_file:
        vehicle_keys = list(input_file.keys())
        total_vehicles = len(vehicle_keys)

    # Determine the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Calculate chunk size based on the number of vehicles and available cores
    chunk_size = max(1, total_vehicles // num_cores)

    # Divide vehicle keys into chunks
    vehicle_chunks = list(chunkify(vehicle_keys, chunk_size))

    # Create a lock using multiprocessing.Manager
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_vehicle_chunk, chunk, input_file_path, output_file_path, lock): chunk for chunk in vehicle_chunks}

        for future in as_completed(futures):
            vehicle_chunk = futures[future]
            try:
                future.result()
                total_number_of_vehicles += len(vehicle_chunk)
                print(f"Processed {total_number_of_vehicles/total_vehicles*100:.2f}% of vehicles", end='\r')
            except Exception as e:
                print(f"Error processing chunk {vehicle_chunk}: {e}")

    print(f"\nTotal vehicles: {total_number_of_vehicles}\n")
    print("Summarized mode durations in windows")


if __name__ == "__main__":

    sum_mode_durations_in_windows()