import pandas as pd
import json
import numpy as np
import h5py
import os
import multiprocessing as mp
import gc
import psutil
import matplotlib.pyplot as plt
import dask.array as da
from collections import Counter
from itertools import combinations
import time
import random

from Data.generate_windows import generate_windows
from Data.summarize_windows import sum_mode_durations_in_windows
from Data.generate_x_and_y import generate_X_and_Y


WINDOW_SIZE = 245 # seconds
DATA_PATH = 'Data/data_with_noise/' # Leave the last slash

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



# ------------------------------- CODE TO PREPROCESS SIMULATION DATA TO USE FOR OTHER PREPROCESSING STEPS -------------------------

@measure_memory_usage
def first_filter(df):

    print("First Filter - Start")
    # for df in df:

    print(df.columns)
    # Filter the lines by only getting the ones where the "type" column is "vector"

    df = df[df['type'] == 'vector']

    # Keep only the columns "module", "name", "vectime", and "vecvalue"
    df = df[['module', 'name', 'vecvalue']]

    # Take the "module" column and use it as the value for a new "vehicle" column
    # And get the integer that is within '[' ']' in the string
    df = df.assign(vehicle=df['module'].str.extract(r'\[(\d+)\]')[0])

    # Drop the "module" column
    df.drop(columns='module', inplace=True)

    # Get lines where name has "tx", "rx", or "idle" on it
    df = df[df['name'].str.contains('tx|rx|idle')]

    # Save to csv
    df.to_csv(f'{DATA_PATH}first_filtered_results.csv', mode='w', index=False)

    # Print the first 5 rows of the df

    # Show all the columns when printing
    pd.set_option('display.max_columns', None)
    
@measure_memory_usage
def second_filter(df):
    print("Second Filter - Start")
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Modify the name column so that it only contain txStart, txEnd, rxStart, rxEnd, idleStart, and idleEnd
    # The current values have txStart, txEnd, rxStart, rxEnd, idleStart, and idleEnd within them
    df['name'] = df['name'].str.extract(r'(txStart|txEnd|rxStart|rxEnd|idleStart|idleEnd)')

    # Define a function to clean vecvalue
    def clean_vecvalue(value):
        # Check if the string has both single and double quotes
        if "'" in value and '"' in value:
            # If it has both, remove single quotes and keep double quotes
            return value.replace("'", '')
        else:
            # Otherwise, just strip any leading or trailing quotes and whitespace
            return value.strip('"\' ')

    # Apply the clean_vecvalue function to vecvalue column
    df['vecvalue'] = df['vecvalue'].apply(clean_vecvalue)

    # Load the JSON file with the vehicle start and end times
    with open(f"{DATA_PATH}vehicle_start_end.json", "r") as file:
        vehicle_start_end = json.load(file)

    # get the number of vehicles
    vehicles = df['vehicle'].unique()
    processed = 0

    # For each vehicle, get the rxStart, rxEnd, idleStart, and idleEnd values
    grouped_df = df.groupby('vehicle')

    vehicles_to_remove = set()
    for vehicle, group in grouped_df:

        vehicle = str(vehicle)
        # If the vehicle has more than 6 lines, skip it and remove it from the df
        if len(group) > 6:
            print(f"Vehicle {vehicle} has more than 6 lines")
            df = df[df['vehicle'] != vehicle]
            continue

        # Get the vecvalues for idleStart, and idleEnd if they exist

        idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
        idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values

        # If any of the values are empty, print a message and skip the vehicle
        if len(idle_start) == 0 or len(idle_end) == 0:
            print(f"Vehicle {vehicle} does not have idleStart or idleEnd values")
            print(f"Idle Start: {idle_start}")
            print(f"Idle End: {idle_end}")
            vehicles_to_remove.add(int(vehicle))
            continue
        
        duration_in_simulation = vehicle_start_end[vehicle]['end'] - vehicle_start_end[vehicle]['start']
        if duration_in_simulation < WINDOW_SIZE:
            print(f"Vehicle {vehicle} took less than window size ({WINDOW_SIZE}s) to complete the simulation")
            vehicles_to_remove.add(int(vehicle))
            continue
    
        # These values are strings in the format 'float float float ... float'
        # Convert them to list of float by splitting the string
        # print(idle_start)
        idle_start = list(map(float, idle_start[0].split()))
        idle_end = list(map(float, idle_end[0].split()))

        # Insert the start time of the vehicle to the first idle_start and the end time of the vehicle to the last idle_end
        idle_start.insert(0, vehicle_start_end[vehicle]['start'])
        idle_end.insert(len(idle_end), vehicle_start_end[vehicle]['end'])

        # Modify the vecvalue column to contain the new values as string in the format 'float float float ... float'
        new_idle_start = ' '.join(map(str, idle_start))
        new_idle_end = ' '.join(map(str, idle_end))

        vehicle = int(vehicle)
        # Update the vecvalue column with the new values
        df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleStart'), 'vecvalue'] = new_idle_start
        df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleEnd'), 'vecvalue'] = new_idle_end

        processed += 1

        # Print the percentage of vehicles processed
        print(f"Second Filter - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')

    # Remove vehicles on the vehicles_to_remove set
    df = df[~df['vehicle'].isin(vehicles_to_remove)]

    # Save to csv
    df.to_csv(f'{DATA_PATH}second_filtered_results.csv', index=False)
    
    print("Second Filter - Done")

@measure_memory_usage
def test_disparities(df):

    df = df.copy()

    def convert_vecvalue(value):
        if value == 'vecvalue':
            return []
        return list(map(float, value.split()))
    
    df['vecvalue'] = df['vecvalue'].apply(convert_vecvalue)
 

    # Count the number of different vehicles in the df
    vehicles = df['vehicle'].unique()
    print(f"Number of vehicles: {len(vehicles)}")

    # Print the number of lines in the df
    print(f"Number of lines: {len(df)}")

    print("Testing disparities")
    # Group df by vehicle
    grouped_df = df.groupby('vehicle')

    count_of_differents = 0

    # Iterate over each vehicle group
    for vehicle, group in grouped_df:
        has_different_lengths = False

        # Extract lengths of 'vecvalue' for each relevant 'name'
        tx_start_len = group.loc[group['name'] == 'txStart', 'vecvalue'].map(len).values
        tx_end_len = group.loc[group['name'] == 'txEnd', 'vecvalue'].map(len).values
        rx_start_len = group.loc[group['name'] == 'rxStart', 'vecvalue'].map(len).values
        rx_end_len = group.loc[group['name'] == 'rxEnd', 'vecvalue'].map(len).values
        idle_start_len = group.loc[group['name'] == 'idleStart', 'vecvalue'].map(len).values
        idle_end_len = group.loc[group['name'] == 'idleEnd', 'vecvalue'].map(len).values

        # Compare lengths where applicable
        if len(tx_start_len) > 0 and len(tx_end_len) > 0 and tx_start_len[0] != tx_end_len[0]:
            has_different_lengths = True
            print(f"Vehicle {vehicle} has a difference in length between txStart and txEnd")
        
        if len(rx_start_len) > 0 and len(rx_end_len) > 0 and rx_start_len[0] != rx_end_len[0]:
            has_different_lengths = True
            print(f"Vehicle {vehicle} has a difference in length between rxStart and rxEnd")
        
        if len(idle_start_len) > 0 and len(idle_end_len) > 0 and idle_start_len[0] != idle_end_len[0]:
            has_different_lengths = True
            print(f"Vehicle {vehicle} has a difference in length between idleStart and idleEnd")

        if has_different_lengths:
            count_of_differents += 1

        print(".", end='')

    # Print the number of vehicles with different lengths
    print(f"\nNumber of vehicles with different lengths: {count_of_differents}")

    # Print the number of vehicles with equal lengths
    print(f"Number of vehicles with equal lengths: {len(vehicles) - count_of_differents}")

    return count_of_differents, len(vehicles) - count_of_differents

@measure_memory_usage
def remove_disparities(df):
    df_original = df.copy()
    df = df.copy()

    def convert_vecvalue(value):
        if value == 'vecvalue':
            return []
        return list(map(float, value.split()))
    
    df['vecvalue'] = df['vecvalue'].apply(convert_vecvalue)

    print("Removing disparities")

    # Group df by vehicle
    grouped_df = df.groupby('vehicle')

    # Identify vehicles with different lengths between txStart and txEnd, rxStart and rxEnd, idleStart and idleEnd
    vehicles_to_remove = set()

    for vehicle, group in grouped_df:
        tx_start_len = group.loc[group['name'] == 'txStart', 'vecvalue'].map(len).values
        tx_end_len = group.loc[group['name'] == 'txEnd', 'vecvalue'].map(len).values
        rx_start_len = group.loc[group['name'] == 'rxStart', 'vecvalue'].map(len).values
        rx_end_len = group.loc[group['name'] == 'rxEnd', 'vecvalue'].map(len).values
        idle_start_len = group.loc[group['name'] == 'idleStart', 'vecvalue'].map(len).values
        idle_end_len = group.loc[group['name'] == 'idleEnd', 'vecvalue'].map(len).values

        if len(tx_start_len) > 0 and len(tx_end_len) > 0 and tx_start_len[0] != tx_end_len[0]:
            vehicles_to_remove.add(vehicle)
        
        if len(rx_start_len) > 0 and len(rx_end_len) > 0 and rx_start_len[0] != rx_end_len[0]:
            vehicles_to_remove.add(vehicle)
        
        if len(idle_start_len) > 0 and len(idle_end_len) > 0 and idle_start_len[0] != idle_end_len[0]:
            vehicles_to_remove.add(vehicle)

        print(".", end='')

    print("\nVehicles with disparities:", vehicles_to_remove)

    # Remove vehicles with disparities
    df_original = df_original[~df_original['vehicle'].isin(vehicles_to_remove)]

    # Save to csv
    df_original.to_csv(f'{DATA_PATH}no_disparity_filtered_results.csv', index=False)

    return df_original

@measure_memory_usage
def separate_rx_idle(df):

    df = df.copy()
    
    df = df[df['vehicle'] != 'vehicle']

    # For each vehicle, get the rxStart, rxEnd, idleStart, and idleEnd values
    grouped_df = df.groupby('vehicle')

    # Get the number of vehicles
    vehicles = df['vehicle'].unique()

    processed = 0
    for vehicle, group in grouped_df:

        # Get the vecvalues for rxStart, rxEnd, idleStart, and idleEnd if they exist
        rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
        rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
        idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
        idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values

        # If any of the values are empty, skip the vehicle
        if len(rx_start) == 0 or len(rx_end) == 0 or len(idle_start) == 0 or len(idle_end) == 0:
            continue

        # These values are strings in the format 'float float float ... float'
        # Convert them to list of float by splitting the string
        rx_start = list(map(float, rx_start[0].split()))
        rx_end = list(map(float, rx_end[0].split()))
        idle_start = list(map(float, idle_start[0].split()))
        idle_end = list(map(float, idle_end[0].split()))

        new_idle_start = idle_start.copy()
        new_idle_end = idle_end.copy()

        # Temporary lists to hold the new values
        temp_idle_start = []
        temp_idle_end = []

        for i in range(len(rx_start)):
            for j in range(len(idle_start)):
                if rx_start[i] >= idle_start[j] and rx_end[i] <= idle_end[j]:
                    # Add the split parts to the temporary lists
                    temp_idle_start.append((j+1, rx_end[i]))
                    temp_idle_end.append((j, rx_start[i]))
                    break

        # Insert the new intervals into the new_idle_start and new_idle_end lists
        for idx, value in sorted(temp_idle_start, reverse=True):
            new_idle_start.insert(idx, value)
        for idx, value in sorted(temp_idle_end, reverse=True):
            new_idle_end.insert(idx, value)

        # Modify the vecvalue column to contain the new values as string in the format 'float float float ... float'
        new_idle_start = ' '.join(map(str, new_idle_start))
        new_idle_end = ' '.join(map(str, new_idle_end))

        # Update the vecvalue column with the new values
        df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleStart'), 'vecvalue'] = new_idle_start
        df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleEnd'), 'vecvalue'] = new_idle_end

        processed += 1

        """ # Print the vecvalue of idleStart and idleEnd of the df
        print("-------------------")
        print("vehicle", vehicle)
        print('start', len(df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleStart'), 'vecvalue'].values[0].split()))
        print('end', len(df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleEnd'), 'vecvalue'].values[0].split())) """
       
        # Print the percentage of vehicles processed
        print(f"Separate RX and IDLE - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')   

    # Print the vecvalue of idleStart and idleEnd of the df for each vehicle
    """ print("-------------------")
    grouped_df = df.groupby('vehicle')

    count = 0
    for vehicle, group in grouped_df:
        try:
            print("vehicle", vehicle)
            print('start', len(group.loc[group['name'] == 'idleStart', 'vecvalue'].values[0].split()))
            print('end', len(group.loc[group['name'] == 'idleEnd', 'vecvalue'].values[0].split()))
        except:
            print("vehicle fail", vehicle)

        count += 1
        if count == 11:

            break """


    # Save to csv
    df.to_csv(f'{DATA_PATH}separated_rx_idle_results.csv', index=False)

    print("Separated rx and idle")
    return df

# -------------------------------------------------------------------------------------------------------------------------

###### THE CODE BELOW HAS BEEN CLEANED AND IMPROVED AND SEPARATED INTO OTHER FILES. ######
###### IT WAS KEPT AS A REFERENCE OF WHAT HAS BEEN MADE. ######

# FIRST VERSION OF THE CODE TO CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES
""" @measure_memory_usage
def calculate_optimal_window_size_by_switches(df) -> int:

    def sturges(n: int):
        result = np.ceil(np.log2(n))
        return int(result) + 1
    

    df = df.copy()

    df = df[df['vehicle'] != 'vehicle']

    vehicles = df['vehicle'].unique()

    downsample_factor = 1000
    # Get only two vehicles to debug
    df = df[df['vehicle'].isin(random.sample(range(len(vehicles)), len(vehicles)//downsample_factor))]
    grouped_df = df.groupby('vehicle')

    vehicles = df['vehicle'].unique()

    processed = 0

    optimal_window_sizes = []
    mean_time_intervals = []

    window_sizes = range(30, 2001, 100)
    
    for vehicle, group in grouped_df:
        start_time = time.time()
        # Get the vecvalues for rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd if they exist
        rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
        rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
        idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
        idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values
        tx_start = group.loc[group['name'] == 'txStart', 'vecvalue'].values
        tx_end = group.loc[group['name'] == 'txEnd', 'vecvalue'].values
        
        # Convert them to list of float by splitting the string if they exist
        rx_start = list(map(float, rx_start[0].split())) if len(rx_start) > 0 else []
        rx_end = list(map(float, rx_end[0].split())) if len(rx_end) > 0 else []
        idle_start = list(map(float, idle_start[0].split())) if len(idle_start) > 0 else []
        idle_end = list(map(float, idle_end[0].split())) if len(idle_end) > 0 else []
        tx_start = list(map(float, tx_start[0].split())) if len(tx_start) > 0 else []
        tx_end = list(map(float, tx_end[0].split())) if len(tx_end) > 0 else []

        # Combine _start data into a single list of tuples (mode, time)
        start = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) 
                for mode, time in zip(['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start), 
                                    rx_start + idle_start + tx_start)]

        # Sort data by time and save the new order to apply to _end data
        index_order = np.argsort([time for mode, time in start])
        start = [start[i] for i in index_order]

        # Combine _end data into a single list of tuples (mode, time)
        end = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) 
            for mode, time in zip(['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end), 
                                    rx_end + idle_end + tx_end)]

        # Sort data by time using the order from the _start data
        end = [end[i] for i in index_order]

        # Convert sorted data back to NumPy arrays
        start = np.array(start)
        end = np.array(end)

        # Calculate durations by subtracting the second column of end from the second column of start but keep the mode
        durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

        durations_df = pd.DataFrame(durations, columns=['mode', 'duration'])
        durations_df['duration'] = durations_df['duration'].apply(float)

        durations = list(durations_df.to_records(index=False))

        variance_scores = evaluate_window_sizes(durations_df, window_sizes)
        best_window_size = find_optimal_window_size(variance_scores, window_sizes)
        optimal_window_sizes.append(best_window_size)

        # Calculate the average time interval for the optimal window size
        optimal_window_durations = durations_df['duration'].rolling(window=best_window_size).sum().dropna()
        mean_time_interval_ms = optimal_window_durations.mean()
        mean_time_intervals.append(mean_time_interval_ms)

        processed += 1
        print(f"Calculating Optimal Window Size - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')

        print(f"Time taken for vehicle: {time.time() - start_time}", flush=True)

    # Calculate the mean optimal window size across all vehicles
    mean_optimal_window_size = np.mean(optimal_window_sizes)
    print(f"Mean optimal window size across all vehicles (in number of observations): {mean_optimal_window_size}")

    mean_time_intervals = np.nan_to_num(mean_time_intervals)
    # Calculate the mean time interval across all vehicles for the mean optimal window size
    mean_mean_time_interval_ms = np.mean(mean_time_intervals)
    print(f"Mean time interval across all vehicles for mean optimal window size in milliseconds: {mean_mean_time_interval_ms:.2f}")

    print(f"Variance time interval across all vehicles for mean optimal window size in milliseconds: {np.std(mean_time_intervals)}")
    
    
    mean_value = np.mean(mean_time_intervals)
    median_value = np.median(mean_time_intervals)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_time_intervals, bins=sturges(len(vehicles)), color='skyblue', edgecolor='black')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.title('Histogram of Mean Time Intervals')
    plt.xlabel('Mean Time Interval (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{DATA_PATH}histogram_mean_optimal_time_window_by_switches.png')

    return mean_mean_time_interval_ms
"""


""" 

# ---------------- HELPER FUNCTIONS FOR THE CODES THAT CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES ------------------------

def count_mode_switches(df, window_size):
    switch_counts = []
    modes = df['mode'].values
    for i in range(len(modes) - window_size + 1):
        window = modes[i:i + window_size]
        window_switch_counts = {}
        for mode1, mode2 in combinations(set(window), 2):
            count = sum((window[j] == mode1 and window[j + 1] == mode2) or
                        (window[j] == mode2 and window[j + 1] == mode1)
                        for j in range(len(window) - 1))
            window_switch_counts[(mode1, mode2)] = count
        switch_counts.append(window_switch_counts)
    return switch_counts

def evaluate_window_sizes(df, window_sizes):
    variance_scores = []
    processed = 0
    for window_size in window_sizes:
        switch_counts = count_mode_switches(df, window_size)
        switch_counts_flat = [switch_count for window_switch_counts in switch_counts for switch_count in window_switch_counts.values()]
        variance_score = np.var(switch_counts_flat)
        variance_scores.append(variance_score)
        processed += 1
        #print(f'Evaluated {processed*100/len(window_sizes)}% window sizes for the vehicle', end='\r', flush=True)
    return variance_scores

def find_optimal_window_size(variance_scores, window_sizes):
    best_window_size = window_sizes[np.argmin(variance_scores)]
    return best_window_size


# ----------------- END HELPER FUNCTIONS -----------------------------------------

# ---------------------------- PARALLEL CODE TO CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES --------------------------------------------------

def sturges(n: int):
    result = np.ceil(np.log2(n))
    return int(result) + 1

def process_vehicle(group, window_sizes):
    start_time = time.time()
    # Get the vecvalues for rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd if they exist
    rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
    rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
    idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
    idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values
    tx_start = group.loc[group['name'] == 'txStart', 'vecvalue'].values
    tx_end = group.loc[group['name'] == 'txEnd', 'vecvalue'].values
    
    # Convert them to list of float by splitting the string if they exist
    rx_start = list(map(float, rx_start[0].split())) if len(rx_start) > 0 else []
    rx_end = list(map(float, rx_end[0].split())) if len(rx_end) > 0 else []
    idle_start = list(map(float, idle_start[0].split())) if len(idle_start) > 0 else []
    idle_end = list(map(float, idle_end[0].split())) if len(idle_end) > 0 else []
    tx_start = list(map(float, tx_start[0].split())) if len(tx_start) > 0 else []
    tx_end = list(map(float, tx_end[0].split())) if len(tx_end) > 0 else []

    # Combine _start data into a single list of tuples (mode, time)
    start = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) 
            for mode, time in zip(['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start), 
                                rx_start + idle_start + tx_start)]

    # Sort data by time and save the new order to apply to _end data
    index_order = np.argsort([time for mode, time in start])
    start = [start[i] for i in index_order]

    # Combine _end data into a single list of tuples (mode, time)
    end = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) 
        for mode, time in zip(['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end), 
                                rx_end + idle_end + tx_end)]

    # Sort data by time using the order from the _start data
    end = [end[i] for i in index_order]

    # Convert sorted data back to NumPy arrays
    start = np.array(start)
    end = np.array(end)

    # Calculate durations by subtracting the second column of end from the second column of start but keep the mode
    durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

    durations_df = pd.DataFrame(durations, columns=['mode', 'duration'])
    durations_df['duration'] = durations_df['duration'].apply(float)

    durations = list(durations_df.to_records(index=False))

    variance_scores = evaluate_window_sizes(durations_df, window_sizes)
    best_window_size = find_optimal_window_size(variance_scores, window_sizes)
    #optimal_window_sizes.append(best_window_size)

    # Calculate the average time interval for the optimal window size
    optimal_window_durations = durations_df['duration'].rolling(window=best_window_size).sum().dropna()
    mean_time_interval_ms = optimal_window_durations.mean()
    #mean_time_intervals.append(mean_time_interval_ms)

    #print(f"Calculating Optimal Window Size - Processed {vehicle/len(vehicles)*100:.2f}% of vehicles", end='\r')
    print(f"Time taken for vehicle: {time.time() - start_time}", flush=True)

    return mean_time_interval_ms

def calculate_optimal_window_size_by_switches(df) -> int:

    df = df.copy()

    df = df[df['vehicle'] != 'vehicle']

    vehicles = df['vehicle'].unique()

    downsample_factor = 20
    # Get only two vehicles to debug
    df = df[df['vehicle'].isin(random.sample(range(len(vehicles)), len(vehicles)//downsample_factor))]
    grouped_df = df.groupby('vehicle')

    vehicles = df['vehicle'].unique()

    processed = 0

    #optimal_window_sizes = []
    mean_time_intervals = []

    window_sizes = range(30, 2001, 100)

    # Use multiprocessing Pool to parallelize vehicle processing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [pool.apply_async(process_vehicle, args=(group, window_sizes)) for vehicle, group in grouped_df]

        # Ensure all processes are complete
        for result in results:
            mean_time_interval_ms = result.get()
            mean_time_intervals.append(mean_time_interval_ms)
            processed += 1
            print(f"Calculating Optimal Window Size - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')


    # Calculate the mean optimal window size across all vehicles
    #mean_optimal_window_size = np.mean(optimal_window_sizes)
    #print(f"\nMean optimal window size across all vehicles (in number of observations): {mean_optimal_window_size}")

    mean_time_intervals = np.nan_to_num(mean_time_intervals)
    # Calculate the mean time interval across all vehicles for the mean optimal window size
    mean_mean_time_interval_ms = np.mean(mean_time_intervals)
    print(f"Mean time interval across all vehicles for mean optimal window size in milliseconds: {mean_mean_time_interval_ms:.2f}")

    print(f"Variance time interval across all vehicles for mean optimal window size in milliseconds: {np.std(mean_time_intervals)}")
    
    mean_value = np.mean(mean_time_intervals)
    median_value = np.median(mean_time_intervals)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_time_intervals, bins=sturges(len(vehicles)), color='skyblue', edgecolor='black')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.title('Histogram of Mean Time Intervals')
    plt.xlabel('Mean Time Interval (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{DATA_PATH}histogram_mean_optimal_time_window_by_switches.png')

    return mean_mean_time_interval_ms


# ---------------------------- END OF THE PARALELL CODE TO CALCULATE OPTIMAL WINDOW SIZE BY SWITCHES --------------------
 """


# CODE TO CALCULATE OPTIMAL WINDOW SIZE BASED ON VARIANCE CRITERIA - NOT UTILIZED.
""" 
@measure_memory_usage
def calculate_optimal_window_size(df) -> int:
    def sturges(n: int):
        result = np.ceil(np.log2(n))
        return int(result)

    df = df.copy()

    df = df[df['vehicle'] != 'vehicle']

    vehicles = df['vehicle'].unique()



    downsample_factor = 100
    # Get only two vehicles to debug
    df = df[df['vehicle'].isin(random.sample(range(len(vehicles)), len(vehicles)//downsample_factor))]

    grouped_df = df.groupby('vehicle')

    vehicles = df['vehicle'].unique()

    processed = 0

    # Assuming `grouped_df` is your grouped DataFrame
    optimal_window_sizes = []
    mean_time_intervals = []

    for vehicle, group in grouped_df:
        start_time = time.time()
        # Get the vecvalues for rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd if they exist
        rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
        rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
        idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
        idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values
        tx_start = group.loc[group['name'] == 'txStart', 'vecvalue'].values
        tx_end = group.loc[group['name'] == 'txEnd', 'vecvalue'].values
        
        # Convert them to list of float by splitting the string if they exist
        rx_start = list(map(float, rx_start[0].split())) if len(rx_start) > 0 else []
        rx_end = list(map(float, rx_end[0].split())) if len(rx_end) > 0 else []
        idle_start = list(map(float, idle_start[0].split())) if len(idle_start) > 0 else []
        idle_end = list(map(float, idle_end[0].split())) if len(idle_end) > 0 else []
        tx_start = list(map(float, tx_start[0].split())) if len(tx_start) > 0 else []
        tx_end = list(map(float, tx_end[0].split())) if len(tx_end) > 0 else []

        # Combine _start data into a single list of tuples (mode, time)
        start = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) 
                for mode, time in zip(['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start), 
                                    rx_start + idle_start + tx_start)]

        # Sort data by time and save the new order to apply to _end data
        index_order = np.argsort([time for mode, time in start])
        start = [start[i] for i in index_order]

        # Combine _end data into a single list of tuples (mode, time)
        end = [(mode, time * 1000) if mode != 'tx' else (mode, time / 1000) 
            for mode, time in zip(['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end), 
                                    rx_end + idle_end + tx_end)]

        # Sort data by time using the order from the _start data
        end = [end[i] for i in index_order]

        # Convert sorted data back to NumPy arrays
        start = np.array(start)
        end = np.array(end)

        # Calculate durations by subtracting the second column of end from the second column of start but keep the mode
        durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

        # Normalize durations based on mode
        durations_df = pd.DataFrame(durations, columns=['mode', 'duration'])

        durations_df['duration'] = durations_df['duration'].apply(float)
  
        # print(durations_df.head())
        # print(durations_df.dtypes)
        #print(durations_df.groupby('mode')['duration'].mean().head())
        #print(durations_df.dtypes)
        mode_means = durations_df.groupby('mode')['duration'].mean()
        mode_stds = durations_df.groupby('mode')['duration'].std()
        #print(mode_means)
        #print(mode_stds)
        def normalize(row):
            return (row['duration'] - mode_means[row['mode']]) / mode_stds[row['mode']]
        
        durations_df['normalized_duration'] = durations_df.apply(normalize, axis=1)

        # print(durations_df.head())
        # Function to calculate rolling variance with different window sizes
        def calculate_rolling_variance(df, window_sizes):
            variances = []
            for window_size in window_sizes:
                rolling_var = df['normalized_duration'].rolling(window=window_size).var().dropna()
                mean_var = rolling_var.mean()
                variances.append(mean_var)
            return variances

        # Define the range of window sizes to evaluate
        window_sizes = range(30, 2001, 100)  # Example: window sizes from 2 to 20

        # Calculate rolling variances
        variances = calculate_rolling_variance(durations_df, window_sizes)

        # Find the optimal window size (minimize mean rolling variance)
        best_window_size = window_sizes[np.argmin(variances)]
        optimal_window_sizes.append(best_window_size)

        #print(f"Vehicle {vehicle}: Best window size based on variance stabilization = {best_window_size}")

        # Calculate the average time interval for the optimal window size
        optimal_window_durations = durations_df['duration'].rolling(window=best_window_size).sum().dropna()
        mean_time_interval_ms = optimal_window_durations.mean()
        mean_time_intervals.append(mean_time_interval_ms)

        #print(f"Mean time interval for optimal window size in milliseconds: {mean_time_interval_ms:.2f}")

        # Optional: Plot the rolling variance for the current vehicle
        plt.figure(figsize=(10, 6))
        plt.plot(window_sizes, variances, marker='o', linestyle='-')
        plt.xlabel('Window Size')
        plt.ylabel('Mean Rolling Variance')
        plt.title(f'Mean Rolling Variance vs. Window Size for Vehicle {vehicle}')
        plt.grid(True)
        plt.show()

        print(f"Time taken for vehicle: {time.time() - start_time}", flush=True)
        processed += 1
        print(f"Calculating Optimal Window Size - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')

    # Calculate the mean optimal window size across all vehicles
    mean_optimal_window_size = np.mean(optimal_window_sizes)
    print(f"Mean optimal window size across all vehicles (in number of observations): {mean_optimal_window_size}")

    mean_time_intervals = np.nan_to_num(mean_time_intervals)
    
    # Plot the histogram
    mean_value = np.mean(mean_time_intervals)
    median_value = np.median(mean_time_intervals)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_time_intervals, bins=sturges(len(vehicles)), color='skyblue', edgecolor='black')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.title('Histogram of Mean Time Intervals')
    plt.xlabel('Mean Time Interval (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{DATA_PATH}histogram_mean_optimal_time_window_by_variance.png')
        
    # Calculate the mean time interval across all vehicles for the mean optimal window size
    mean_mean_time_interval_ms = np.mean(mean_time_intervals)
    print(f"Mean time interval across all vehicles for mean optimal window size in milliseconds: {mean_mean_time_interval_ms:.2f}")

    # print(mean_time_intervals)
    print(f"Variance time interval across all vehicles for mean optimal window size in milliseconds: {np.std(mean_time_intervals)}")
    

    return mean_mean_time_interval_ms

 """
# FIRST VERSION OF GENERATE WINDOWS

""" @measure_memory_usage
def generate_windows(df, window_size_in_seconds=WINDOW_SIZE):

    window_size_in_ms = 245 # !!!!!

    df = df.copy()

    df = df[df['vehicle'] != 'vehicle']

    vehicles = df['vehicle'].unique()
    # Filter to only get the vehicle 2949 and 2950 FOR DEBUG
    #df = df[df['vehicle'].isin([1, 800])]

    downsample_factor = 10
    # Get only two vehicles to debug
    df = df[df['vehicle'].isin(random.sample(range(len(vehicles)), len(vehicles)//downsample_factor))]

    # For each vehicle get the rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd values
    grouped_df = df.groupby('vehicle')

    # Get the number of vehicles
    vehicles = df['vehicle'].unique()
    processed = 0

    # Create a list to store the windows for each vehicle
    all_windows = []

    rx_durations = []
    tx_durations = []
    idle_durations = []
    
  
    for vehicle, group in grouped_df:

        # Get the vecvalues for rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd if they exist
        rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
        rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
        idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
        idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values
        tx_start = group.loc[group['name'] == 'txStart', 'vecvalue'].values
        tx_end = group.loc[group['name'] == 'txEnd', 'vecvalue'].values

        # These values are strings in the format 'float float float ... float'
        # Convert them to list of float by splitting the string if they exist
        rx_start = list(map(float, rx_start[0].split())) if len(rx_start) > 0 else []
        rx_end = list(map(float, rx_end[0].split())) if len(rx_end) > 0 else []
        idle_start = list(map(float, idle_start[0].split())) if len(idle_start) > 0 else []
        idle_end = list(map(float, idle_end[0].split())) if len(idle_end) > 0 else []
        tx_start = list(map(float, tx_start[0].split())) if len(tx_start) > 0 else []
        tx_end = list(map(float, tx_end[0].split())) if len(tx_end) > 0 else []


        for i in range(len(rx_start)):
            rx_durations.append(rx_end[i] - rx_start[i])

        for i in range(len(idle_start)):
            idle_durations.append(idle_end[i] - idle_start[i])
        
        for i in range(len(tx_start)):
            tx_durations.append(tx_end[i] - tx_start[i])

        # If any of the durations are negative, break
        if np.any(np.array(rx_durations) < 0) or np.any(np.array(idle_durations) < 0) or np.any(np.array(tx_durations) < 0):
            print(f"Negative duration for vehicle {vehicle}")
            #break

        # Combine _start data into a single list of tuples (mode, time)
        start = [(mode, time*1000) if mode != 'tx' else (mode, time/1000) for mode, time in zip(['rx']*len(rx_start) + ['idle']*len(idle_start) + ['tx']*len(tx_start), rx_start + idle_start + tx_start)]

        # Sort data by time and save the new order to apply to _end data
        index_order = np.argsort([time for mode, time in start])
        start = [start[i] for i in index_order]

        # Combine _end data into a single list of tuples (mode, time)
        end = [(mode, time*1000) if mode != 'tx' else (mode, time/1000) for mode, time in zip(['rx']*len(rx_end) + ['idle']*len(idle_end) + ['tx']*len(tx_end), rx_end + idle_end + tx_end)]

        # Sort data by time using the order from the _start data
        end = [end[i] for i in index_order]

        # Convert sorted data back to NumPy arrays
        start = np.array(start)
        end = np.array(end)

        np.printoptions(precision=20)
        # Calculate durations by subtracting the second column of end from the second column of start but keep the mode
        
        # Durations is the same shape as start and end
        durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

        # For each duration, if the duration is higher than window_size_in_ms, split it into multiple durations so that each duration is less than window_size_in_ms
        i = 0
        while i < len(durations):
            if float(durations[i][1]) > window_size_in_ms:
                # Calculate the number of windows needed
                num_windows = int(np.ceil(float(durations[i][1]) / window_size_in_ms))

                # Calculate the duration of each window
                window_duration = float(durations[i][1]) / num_windows

                # Create the windows
                windows = np.array([[durations[i][0], window_duration]] * num_windows, dtype=object)

                # Insert the windows into the durations array and remove the original duration
                durations = np.insert(durations, i + 1, windows, axis=0)
                durations = np.delete(durations, i, axis=0)

                # Adjust the index to account for the inserted windows
                i += num_windows - 1
            i += 1

        # Check if there is any duration that is higher than window_size_in_ms
        if np.any(durations[:, 1].astype(float) > window_size_in_ms):
            print(f"Duration higher than window size")
     

        # Check if there is any negative 
        try:
            if np.any(durations[:, 1].astype(float) < 0.0):
                print(f"Negative duration for vehicle {vehicle}")
        except:
            print(f"Negative duration for vehicle {vehicle}")


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

        # Print the windows 
        #print(windows_for_vehicle)



        # Append the windows for the vehicle to the all_windows list
        all_windows.append(windows_for_vehicle)

        processed += 1
        print(f"Generate Windows - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')    

    # Create histograms for the durations in subplots
    

    # Plot the distribution of durations in subplots
    fig, axs = plt.subplots(6, 1, figsize=(10, 10))

    axs[0].hist(tx_durations, bins=100, color='blue', log=True)
    axs[0].set_title('TX Durations')
    axs[0].set_xlabel('Duration (ms)')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(rx_durations, bins=100, color='green', log=True)
    axs[1].set_title('RX Durations')
    axs[1].set_xlabel('Duration (ms)')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(idle_durations, bins=100, color='red', log=True)
    axs[2].set_title('Idle Durations')
    axs[2].set_xlabel('Duration (ms)')
    axs[2].set_ylabel('Frequency')


    # Plot histograms for the differences in durations
    axs[3].hist(np.diff(tx_durations), bins=100, color='blue', log=True)
    axs[3].set_title('TX Durations Differences')
    axs[3].set_xlabel('Difference (ms)')
    axs[3].set_ylabel('Frequency')

    axs[4].hist(np.diff(rx_durations), bins=100, color='green', log=True)
    axs[4].set_title('RX Durations Differences')
    axs[4].set_xlabel('Difference (ms)')
    axs[4].set_ylabel('Frequency')

    axs[5].hist(np.diff(idle_durations), bins=100, color='red', log=True)
    axs[5].set_title('Idle Durations Differences')
    axs[5].set_xlabel('Difference (ms)')
    axs[5].set_ylabel('Frequency')
    
    plt.tight_layout()

    plt.savefig(f'{DATA_PATH}raw_durations_histogram.png')

    plt.close()
    # plt.show()

    
    print("Number of vehicles in all_windows", len(all_windows))
    total_number_of_windows = 0

    print("Saving to hdf5 file (this takes a while)")
    # Remove the previous file if it exists
    try:
        os.remove(f'{DATA_PATH}windows{window_size_in_seconds}s.hdf5')
    except:
        pass
    # Save to hdf5 file
    with h5py.File(f'{DATA_PATH}windows{window_size_in_seconds}s.hdf5', 'a') as file:
        for i, vehicle_windows in enumerate(all_windows):
            for j, window in enumerate(vehicle_windows):
                total_number_of_windows += 1

                # Change the type to S32
                window = np.array(window, dtype='S32')
                
                file.create_dataset(f'vehicle_{i}/window_{j}', data=window)

    print(f"Total number of windows: {total_number_of_windows}")

    print("Generated windows")


 """

""" # FIRST VERSION OF THE FUNCTION THAT PLOTS THE HISTOGRAMS OF THE SUMMARIZED DURATIONS

def plot_tx_durations(all_data):

    import matplotlib.pyplot as plt

    all_tx_durations = []

    all_rx_durations = []

    all_idle_durations = []

    # Plot the tx durations for each vehicle
    for vehicle_key, vehicle_data in all_data.items():
        # Get the tx durations
        tx_durations = []

        for window in vehicle_data:
            for j in range(len(window)):
                if window[j][0] == 'tx':
                    all_tx_durations.append(float(window[j][1]))

                if window[j][0] == 'rx':
                    all_rx_durations.append(float(window[j][1]))

                if window[j][0] == 'idle':
                    all_idle_durations.append(float(window[j][1]))


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

    plt.savefig(f'{DATA_PATH}windowed_durations_histogram.png')

# SECOND VERSION OF THE FUNCTION THAT PLOTS THE HISTOGRAMS OF THE SUMMARIZED DURATIONS

def plot_summarized_durations(all_data):

    import matplotlib.pyplot as plt

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

 """
# FIRST VERSION OF SUM_MODE_DURATIONS_IN_WINDOWS:

""" @measure_memory_usage
def sum_mode_durations_in_windows():

    print("Summarizing mode durations in windows")

    with h5py.File(f'{DATA_PATH}windows{WINDOW_SIZE}s.hdf5', 'r') as file:
        # Create a dictionary to store all the data
        all_data = {}
        
        # Iterate through the vehicles
        for vehicle_key in file.keys():
            vehicle_data = []
            
            # Iterate through the windows of each vehicle
            for window_key in file[vehicle_key].keys():
                # Read the dataset
                window_data = file[f'{vehicle_key}/{window_key}'][:]
                
                # Convert the data from bytes to string if needed
                window_data = np.char.decode(window_data, 'utf-8')
                
                # Append to the vehicle's data
                vehicle_data.append(window_data)
            
            # Store the vehicle's data in the dictionary
            all_data[vehicle_key] = vehicle_data

    plot_tx_durations(all_data)
    print("plotted windowed durations")

    processed_vehicles = 0

    for vehicle_key, vehicle_data in all_data.items():

        # For each window, both in X and Y, summarize it to have only 3 values: Total duration in idle, total duration in rx and total duration in tx
        for i in range(len(vehicle_data)):

            idle = 0
            rx = 0
            tx = 0
            for j in range(len(vehicle_data[i])):
                if vehicle_data[i][j][0] == 'idle':
                    idle += float(vehicle_data[i][j][1])
                elif vehicle_data[i][j][0] == 'rx':
                    rx += float(vehicle_data[i][j][1])
                elif vehicle_data[i][j][0] == 'tx':
                    tx += float(vehicle_data[i][j][1])
            
            vehicle_data[i] = [idle, rx, tx]

        # Store the vehicle's data in the dictionary
        all_data[vehicle_key] = vehicle_data

        processed_vehicles += 1

        print(f"Summarize mode durations in windows - Processed {processed_vehicles/len(all_data)*100:.2f}% of vehicles", end='\r')

    # Remove the previous file if it exists
    try:
        os.remove(f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5')
    except:
        pass
    # Save to hdf5 file
    with h5py.File(f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5', 'a') as file:
        for i, vehicle_windows in all_data.items():
            for j, window in enumerate(vehicle_windows):
                # Change the type to S32
                window = np.array(window, dtype='S32')
                
                file.create_dataset(f'vehicle_{i}/window_{j}', data=window)

    print("Summarized mode durations in windows") """


# SECOND VERSION OF SUM_MODE_DURATIONS_IN_WINDOWS:

""" @measure_memory_usage
def sum_mode_durations_in_windows():
    print("Summarizing mode durations in windows")
    total_number_of_vehicles = 0
    # Remove the previous file if it exists
    summarized_file_path = f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5'
    try:
        os.remove(summarized_file_path)
    except FileNotFoundError:
        pass

    with h5py.File(f'{DATA_PATH}windows{WINDOW_SIZE}s.hdf5', 'r') as input_file, \
         h5py.File(summarized_file_path, 'a') as output_file:

        # Get the total number of vehicles
        total_vehicles = len(input_file.keys())
        processed_vehicles = 0

        # Iterate through the vehicles
        for vehicle_key in input_file.keys():
            vehicle_data = []
            
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
                vehicle_data.append(window_data)
            
            # Process the vehicle's data
            for i in range(len(vehicle_data)):
                idle = 0
                rx = 0
                tx = 0
                for j in range(len(vehicle_data[i])):
                    if vehicle_data[i][j][0] == 'idle':
                        idle += float(vehicle_data[i][j][1])
                    elif vehicle_data[i][j][0] == 'rx':
                        rx += float(vehicle_data[i][j][1])
                    elif vehicle_data[i][j][0] == 'tx':
                        tx += float(vehicle_data[i][j][1])
                
                vehicle_data[i] = [idle, rx, tx]

            # Write the processed data to the output file
            for j, window in enumerate(vehicle_data):
                window = np.array(window, dtype='S32')
                output_file.create_dataset(f'vehicle_{vehicle_key}/window_{j}', data=window)

            # Update and print the progress
            processed_vehicles += 1
            total_number_of_vehicles += 1
            print(f"Processed {processed_vehicles/total_vehicles*100:.2f}% of vehicles", end='\r')

    print(f"\nTotal vehicles: {total_number_of_vehicles}\n")
    print("Summarized mode durations in windows") """


# FIRST VERSION OF GENERATE X AND Y:

""" @measure_memory_usage
def generate_X_and_Y(num_past_windows: int = 1):
    print("Generating X and Y data")

    with h5py.File(f'{DATA_PATH}windows{WINDOW_SIZE}s_summarized.hdf5', 'r') as file:
        # Create a dictionary to store all the data
        all_data = {}

        num_vehicles = len(file.keys())
        processed = 0
        # Iterate through the vehicles
        for vehicle_key in file.keys():
            vehicle_data = []
            
            # Iterate through the windows of each vehicle
            for window_key in file[vehicle_key].keys():
                # Read the dataset
                window_data = file[f'{vehicle_key}/{window_key}'][:]
                
                # Convert the data from bytes to string if needed
                window_data = np.char.decode(window_data, 'utf-8').astype(float)
                
                # Append to the vehicle's data
                vehicle_data.append(window_data)
            
            # Store the vehicle's data in the dictionary
            all_data[vehicle_key] = np.array(vehicle_data)
            processed += 1
            print(f"Processed {processed/num_vehicles*100:.2f}% of vehicles", end='\r')

    # Plot summarized windowed durations
    # plot_summarized_durations(all_data)

    # Create the X and Y data
    X, Y = [], []
    
    processed = 0
    # Iterate through the windows of the vehicle and take num_past_windows as X and the next window as Y. Jump by num_past_windows
    for vehicle_key, vehicle_data in all_data.items():
        
        # If the vehicle has less than num_past_windows + 1 windows, skip it
        if len(vehicle_data) < num_past_windows + 1:
            # Print the length of the vehicle's data
            # print(f"Vehicle {vehicle_key} has {len(vehicle_data)} windows, skipping it")
            continue

        # Iterate through the windows of the vehicle and take num_past_windows as X and the next window as Y. Jump by num_past_windows
        for i in range(0, len(vehicle_data) - num_past_windows - 1, num_past_windows + 1):
            if i + num_past_windows < len(vehicle_data):
                X.append(vehicle_data[i:i+num_past_windows])
                # Y.append(vehicle_data[i+num_past_windows])
                idle, rx, _ = vehicle_data[i+num_past_windows]
                Y.append((idle, rx))

        processed += 1
        #print(f"Generate X and Y data - Processed {processed/len(all_data)*100:.2f}% of vehicles", end='\r', flush=True)

    X = np.array(X)
    Y = np.array(Y)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # Save to npy files
    np.save(f'{DATA_PATH}X.npy', X)
    np.save(f'{DATA_PATH}Y.npy', Y)
 """


if __name__ == "__main__":
    # This script preprocesses the simulation data. Uncomment all to do all steps.
    # You can comment steps already done to avoid repeating them.

    # Apply first filter to the raw data
    #raw_file_path = f'{DATA_PATH}results.tar.xz'
    #df = pd.read_csv(raw_file_path, compression='infer', chunksize=10000)
    """ raw_file_path = f'{DATA_PATH}results_cam_sizes.csv'
    df = pd.read_csv(raw_file_path, na_values=['', 'NA'], keep_default_na=True, on_bad_lines='skip', quoting=csv.QUOTE_NONE)
    first_filter(df) """

    # Apply second filter to the first filtered data
    """ first_filtered_path = f"{DATA_PATH}first_filtered_results.csv"
    df = pd.read_csv(first_filtered_path)
    df = reduce_mem_usage(df)
    second_filter(df) """


    # Test disparities
    """ second_filtered_path = f"{DATA_PATH}second_filtered_results.csv"
    df = pd.read_csv(second_filtered_path)
    df = reduce_mem_usage(df)
    vehicles_with_disp_before, vehicles_without_disp_before = test_disparities(df)

    # Remove disparities
    df = reduce_mem_usage(df)
    df = remove_disparities(df)

    # Test disparities again
    vehicles_with_disp_after, vehicles_without_disp_after = test_disparities(df)
    print('\n\n\n\n-----------------------------------')
    print(f"Number of vehicles with different lengths before: {vehicles_with_disp_before}")
    print(f"Number of vehicles with different lengths after: {vehicles_with_disp_after}")

    print(f"Number of vehicles with equal lengths before: {vehicles_without_disp_before}")
    print(f"Number of vehicles with equal lengths after: {vehicles_without_disp_after}") 
     
    
    # Separate rx and idle
    no_disparity_filtered_path = f"{DATA_PATH}no_disparity_filtered_results.csv"
    df = pd.read_csv(no_disparity_filtered_path)
    df = reduce_mem_usage(df)
    df = separate_rx_idle(df)
    
    """ 
   
    # Calculate the optimal window size
    """ separated_rx_idle_path = f"{DATA_PATH}separated_rx_idle_results.csv"
    df = pd.read_csv(separated_rx_idle_path)
    df = reduce_mem_usage(df)
    #optimal_window_size_in_ms = calculate_optimal_window_size(df)
    optimal_window_size_in_ms = calculate_optimal_window_size_by_switches(df) """


    # Generate windows
    """ separated_rx_idle_path = f"{DATA_PATH}separated_rx_idle_results.csv"
    df = pd.read_csv(separated_rx_idle_path)
    df = reduce_mem_usage(df)
    generate_windows(df) """
  
    
    # Summarize mode durations in windows
    # sum_mode_durations_in_windows()
   
    
    # Generate X and Y data
    # generate_X_and_Y(num_past_windows=1)


    