import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import h5py
import gc
import psutil
import random 
WINDOW_SIZE_IN_SECONDS = 14 # seconds
DATA_PATH = 'Data/data_with_noise/' # Leave the last slash

def process_vehicle(vehicle, group, window_size_in_ms):
    #print(f"Processing Vehicles", flush=True)

    rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
    rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
    idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
    idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values
    tx_start = group.loc[group['name'] == 'txStart', 'vecvalue'].values
    tx_end = group.loc[group['name'] == 'txEnd', 'vecvalue'].values

    rx_start = list(map(float, rx_start[0].split())) if len(rx_start) > 0 else []
    rx_end = list(map(float, rx_end[0].split())) if len(rx_end) > 0 else []
    idle_start = list(map(float, idle_start[0].split())) if len(idle_start) > 0 else []
    idle_end = list(map(float, idle_end[0].split())) if len(idle_end) > 0 else []
    tx_start = list(map(float, tx_start[0].split())) if len(tx_start) > 0 else []
    tx_end = list(map(float, tx_end[0].split())) if len(tx_end) > 0 else []

    start = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) for mode, time in zip(['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start), rx_start + idle_start + tx_start)]
    index_order = np.argsort([time for mode, time in start])
    start = [start[i] for i in index_order]

    end = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) for mode, time in zip(['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end), rx_end + idle_end + tx_end)]

    if len(index_order) > len(end):
        raise ValueError(f"Index order length {len(index_order)} is greater than end length {len(end)} for vehicle {vehicle}")

    end = [end[i] for i in index_order]

    start = np.array(start)
    end = np.array(end)

    durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

    i = 0
    while i < len(durations):
        if float(durations[i][1]) > window_size_in_ms:
            num_windows = int(np.ceil(float(durations[i][1]) / window_size_in_ms))
            window_duration = float(durations[i][1]) / num_windows
            windows = np.array([[durations[i][0], window_duration]] * num_windows, dtype=object)
            durations = np.insert(durations, i + 1, windows, axis=0)
            durations = np.delete(durations, i, axis=0)
            i += num_windows - 1
        i += 1

    # ----------------------------- WINDOWING ALGORITHM -----------------------------
    current_duration = 0
    duration_index = 0
    window_index = 0
    windows_for_vehicle = []
    #print(len(durations))

    while duration_index < len(durations) - 1:

        while current_duration < window_size_in_ms:
            
            if duration_index == len(durations):
                #print("End of durations")
                break
            # If the windows list does not have the index window_index, add a new window
            try:
                windows_for_vehicle[window_index]
            except:
                windows_for_vehicle.append([])

            if float(durations[duration_index][1]) + current_duration <= window_size_in_ms:
                windows_for_vehicle[window_index].append(durations[duration_index])
                current_duration += float(durations[duration_index][1])
            else:
                windows_for_vehicle.append([])
                windows_for_vehicle[window_index].append(np.array([durations[duration_index][0], window_size_in_ms - current_duration]))
                windows_for_vehicle[window_index + 1].append(np.array([durations[duration_index][0], float(durations[duration_index][1]) - (window_size_in_ms - current_duration)]))
                current_duration += window_size_in_ms - current_duration

            if current_duration > window_size_in_ms:
                print(f"1 - Vehicle {vehicle} - Window {window_index} - Current duration: {current_duration}")
            duration_index += 1
            # print('.', end='', flush=True)

        else:
            # If the windows list has the index window_index + 1, set the current_duration to the duration of the first element of the next window
            if current_duration > window_size_in_ms:
                try:
                    print(f"2 - Vehicle {vehicle} - Window {window_index} - Current duration: {current_duration} - {windows_for_vehicle[window_index + 1][0][1]}")
                except:
                    print(f"3 - Vehicle {vehicle} - Window {window_index} - Current duration: {current_duration} - NO NEXT WINDOW")

            try:
                current_duration = float(windows_for_vehicle[window_index + 1][0][1])
            except Exception as e:
                #print(e)
                current_duration = 0
            
            # Check if the window duration is equal to the window size
            sum_duration = 0
            for duration in windows_for_vehicle[window_index]:
                sum_duration += float(duration[1])

            #if sum_duration != window_size_in_ms:
                
                #print(f"Vehicle {vehicle} - Window {window_index} - Sum duration: {sum_duration} - Current duration: {current_duration}")


            window_index += 1
    
    else:

        # If the last window does not have the correct duration, remove it
        if sum([float(duration[1]) for duration in windows_for_vehicle[-1]]) != window_size_in_ms:
            windows_for_vehicle.pop(-1)
            #print(f"Vehicle {vehicle} - Last window does not have the correct duration, removed it")


    # Print the number of windows for the vehicle
    #print(f"Number of windows for vehicle {vehicle}: {len(windows_for_vehicle)}")
            
    # ----------------------------- WINDOWING ALGORITHM -----------------------------

    
    return vehicle, windows_for_vehicle

def save_to_hdf5(filename, results, lock):
    with lock:
        with h5py.File(filename, 'a') as file:
            for vehicle, windows_for_vehicle in results:
                for j, window in enumerate(windows_for_vehicle):
                    window = np.array(window, dtype='S32')
                    file.create_dataset(f'vehicle_{vehicle}/window_{j}', data=window)

def process_chunk(vehicle_groups, vehicles, window_size_in_ms, filename, lock):
    #print("Processing Chunk", flush=True)
    results = []
    processed = 0
    for vehicle, group in vehicle_groups:
        try:
            vehicle, windows = process_vehicle(vehicle, group, window_size_in_ms)
            results.append((vehicle, windows))
            processed += 1
            print(f"Generating Chunk Windows - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r', flush=True)
        except Exception as e:
            print(f"Error processing vehicle {vehicle}: {e}")

    
    save_to_hdf5(filename, results, lock)

def generate_windows(df, window_size_in_seconds=WINDOW_SIZE_IN_SECONDS, num_processes=mp.cpu_count()):
    window_size_in_ms = window_size_in_seconds * 1000
    df = df.copy()
    df = df[df['vehicle'] != 'vehicle']
    
    vehicles = df['vehicle'].unique()

    downsample_factor = 1
    # Get only two vehicles to debug
    df = df[df['vehicle'].isin(random.sample(range(len(vehicles)), len(vehicles)//downsample_factor))]

    vehicles = df['vehicle'].unique()

    print(f"Number of vehicles: {len(vehicles)}")
    grouped_df = list(df.groupby('vehicle'))

    manager = mp.Manager()
    lock = manager.Lock()

    filename = f'{DATA_PATH}windows{int(window_size_in_seconds*1000)}s.hdf5'
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    chunk_size = len(grouped_df) // (num_processes*2) if len(grouped_df) // (num_processes*2) > 0 else 1

    chunks = [grouped_df[i:i + chunk_size] for i in range(0, len(grouped_df), chunk_size)]

    # print(f"Chunks: {chunks}", flush=True)

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(process_chunk, [(chunk, vehicles, window_size_in_ms, filename, lock) for chunk in chunks])

    print("Generated windows")

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
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
    if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    gc.collect()
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

if __name__ == "__main__":

    # Generate windows
    separated_rx_idle_path = f"{DATA_PATH}separated_rx_idle_results.csv"
    df = pd.read_csv(separated_rx_idle_path)
    df = reduce_mem_usage(df)
    generate_windows(df)