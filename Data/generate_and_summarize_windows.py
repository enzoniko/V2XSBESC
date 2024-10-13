import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import h5py
import gc
import psutil
import pyarrow.parquet as pq
import pyarrow as pa
import time

WINDOW_SIZE_IN_SECONDS = 5.4  # seconds
DATA_PATH = 'Data/iot/'  # Leave the last slash
SUMMARIZED_WINDOW_SIZE = int(WINDOW_SIZE_IN_SECONDS * 1000)  # in milliseconds

def process_vehicle(vehicle, group, window_size_in_ms):
    start_time = time.time()
    try:
        # Extract values for each event type
        def get_values(name):
            values = group.loc[group['name'] == name, 'vecvalue'].values
            return list(map(float, values[0].split())) if len(values) > 0 else []

        rx_start = get_values('rxStart')
        rx_end = get_values('rxEnd')
        idle_start = get_values('idleStart')
        idle_end = get_values('idleEnd')
        tx_start = get_values('txStart')
        tx_end = get_values('txEnd')

        # Combine start and end times
        start = [(mode, time * 1000) for mode, time in zip(
            ['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start),
            rx_start + idle_start + tx_start
        )]
        index_order = np.argsort([time for mode, time in start])
        start_sorted = [start[i] for i in index_order]

        end = [(mode, time * 1000) for mode, time in zip(
            ['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end),
            rx_end + idle_end + tx_end
        )]

        if len(index_order) > len(end):
            raise ValueError(f"Index order length {len(index_order)} is greater than end length {len(end)} for vehicle {vehicle}")

        end_sorted = [end[i] for i in index_order]

        start_array = np.array(start_sorted)
        end_array = np.array(end_sorted)

        durations = np.column_stack((start_array[:, 0], end_array[:, 1].astype(float) - start_array[:, 1].astype(float)))

        # Adjust windows according to duration
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

        # Windowing algorithm with overflow handling
        current_duration = 0
        duration_index = 0
        window_index = 0
        summarized_windows = []

        idle = 0
        rx = 0
        tx = 0

        while duration_index < len(durations):
            mode, duration = durations[duration_index]
            duration = float(duration)

            while duration > 0:
                # Calculate remaining time in current window
                remaining_window = window_size_in_ms - current_duration

                # Determine how much of the duration fits in the current window
                to_add = min(remaining_window, duration)

                if mode == 'idle':
                    idle += to_add
                elif mode == 'rx':
                    rx += to_add
                elif mode == 'tx':
                    tx += to_add

                current_duration += to_add
                duration -= to_add

                # If the current window is full, summarize it and start a new one
                if current_duration >= window_size_in_ms:
                    summarized_windows.append([idle, rx, tx])
                    idle = rx = tx = 0
                    current_duration = 0

            duration_index += 1

        # Clean up memory
        del start_sorted, end_sorted, start_array, end_array, durations
        gc.collect()

        return vehicle, summarized_windows

    except Exception as e:
        raise e
    finally:
        print(f"Processed vehicle {vehicle} in {time.time() - start_time:.2f} seconds")

def save_summarized_to_hdf5(filename, vehicle, summarized_windows, lock):
    with lock:
        with h5py.File(filename, 'a') as file:
            group_path = f'vehicle_{vehicle}'
            if group_path not in file:
                file.create_group(group_path)

            for j, window in enumerate(summarized_windows):
                window = np.array(window, dtype='S32')
                window_path = f'window_{j}'
                file.create_dataset(f'{group_path}/{window_path}', data=window)


def process_chunk(chunk, window_size_in_ms, filename, lock):
    results = []
    processed = 0
    total = len(chunk)

    for vehicle, group in chunk:
        try:
            vehicle, summarized_windows = process_vehicle(vehicle, group, window_size_in_ms)
            save_summarized_to_hdf5(filename, vehicle, summarized_windows, lock)
            processed += 1
            if processed % 100 == 0 or processed == total:
                print(f"Processed {processed}/{total} vehicles", end='\r', flush=True)
        except Exception as e:
            print(f"Error processing vehicle {vehicle}: {e}")

    # Clean up memory
    del results, chunk
    gc.collect()


def generate_and_summarize_windows_from_parquet(parquet_path, window_size_in_seconds=WINDOW_SIZE_IN_SECONDS, num_processes=1):
    window_size_in_ms = window_size_in_seconds * 1000
    manager = mp.Manager()
    lock = manager.Lock()

    filename = f'{DATA_PATH}windows_summarized_{int(window_size_in_seconds * 1000)}ms.hdf5'
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    pool = mp.Pool(processes=num_processes)
    buffered_data = pd.DataFrame()
    batch_size = 100

    parquet_file = pq.ParquetFile(parquet_path)

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()

            # Concatenate with buffered data
            buffered_data = pd.concat([buffered_data, batch_df], ignore_index=True)

            # Identify complete vehicles (with 8 rows)
            vehicle_counts = buffered_data['vehicle'].value_counts()
            complete_vehicles = vehicle_counts[vehicle_counts == 8].index.tolist()

            # Group complete vehicles
            complete_groups = list(buffered_data[buffered_data['vehicle'].isin(complete_vehicles)].groupby('vehicle'))

            # Update buffer by removing complete vehicles
            buffered_data = buffered_data[~buffered_data['vehicle'].isin(complete_vehicles)]

            if complete_groups:
                # Split into chunks for parallel processing
                chunk_size = max(1, len(complete_groups) // (num_processes))
                chunks = [complete_groups[i:i + chunk_size] for i in range(0, len(complete_groups), chunk_size)]

                for chunk in chunks:
                    pool.apply_async(process_chunk, args=(chunk, window_size_in_ms, filename, lock))

        # Process remaining vehicles
        if not buffered_data.empty:
            remaining_groups = list(buffered_data.groupby('vehicle'))
            chunk_size = max(1, len(remaining_groups) // (num_processes))
            chunks = [remaining_groups[i:i + chunk_size] for i in range(0, len(remaining_groups), chunk_size)]

            for chunk in chunks:
                pool.apply_async(process_chunk, args=(chunk, window_size_in_ms, filename, lock))

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        pool.close()
        pool.join()

    print("\nGenerated and summarized windows")


if __name__ == "__main__":
    parquet_path = f"{DATA_PATH}combined.parquet"
    num_processes = mp.cpu_count()

    print("Starting window generation and summarization from Parquet file...")
    generate_and_summarize_windows_from_parquet(parquet_path, WINDOW_SIZE_IN_SECONDS, num_processes)
    print("Window generation and summarization completed.")
