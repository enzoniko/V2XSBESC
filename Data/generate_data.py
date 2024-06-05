import pandas as pd
import json
import numpy as np
import h5py
import os
import multiprocessing
import gc
import psutil

WINDOW_SIZE = 10 # seconds

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

def first_filter(df):

    for chunk in df:

        # Filter the lines by only getting the ones where the "type" column is "vector"

        chunk = chunk[chunk['type'] == 'vector']

        # Keep only the columns "module", "name", "vectime", and "vecvalue"
        chunk = chunk[['module', 'name', 'vectime', 'vecvalue']]

        # Take the "module" column and use it as the value for a new "vehicle" column
        # And get the integer that is within '[' ']' in the string
        chunk['vehicle'] = chunk['module'].str.extract(r'\[(\d+)\]')

        # Drop the "module" column
        chunk.drop(columns='module', inplace=True)

        # Get lines where name has "tx", "rx", or "idle" on it
        chunk = chunk[chunk['name'].str.contains('tx|rx|idle')]

        # Save to csv
        chunk.to_csv('first_filtered_results.csv', mode='a', index=False)

        # Print the first 5 rows of the chunk

        # Show all the columns when printing
        pd.set_option('display.max_columns', None)
    
def second_filter(df):

    # Drop the "vectime" column
    df.drop(columns='vectime', inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Modify the name column so that it only contain txStart, txEnd, rxStart, rxEnd, idleStart, and idleEnd
    # The current values have txStart, txEnd, rxStart, rxEnd, idleStart, and idleEnd within them
    df['name'] = df['name'].str.extract(r'(txStart|txEnd|rxStart|rxEnd|idleStart|idleEnd)')

    # Load the JSON file with the vehicle start and end times
    with open("vehicle_start_end.json", "r") as file:
        vehicle_start_end = json.load(file)

    # get the number of vehicles
    vehicles = df['vehicle'].unique()
    processed = 0

    # For each vehicle, get the rxStart, rxEnd, idleStart, and idleEnd values
    grouped_df = df.groupby('vehicle')

    for vehicle, group in grouped_df:

        # Get the vecvalues for idleStart, and idleEnd if they exist

        idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
        idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values

        # If any of the values are empty, print a message and skip the vehicle
        if len(idle_start) == 0 or len(idle_end) == 0:
            print(f"Vehicle {vehicle} does not have idleStart or idleEnd values")
            print(f"Idle Start: {idle_start}")
            print(f"Idle End: {idle_end}")
            continue
        
        duration_in_simulation = vehicle_start_end[vehicle]['end'] - vehicle_start_end[vehicle]['start']
        if duration_in_simulation < WINDOW_SIZE:
            print(f"Vehicle {vehicle} took less than window size ({WINDOW_SIZE}s) to complete the simulation")
            continue
    
        # These values are strings in the format 'float float float ... float'
        # Convert them to list of float by splitting the string
        idle_start = list(map(float, idle_start[0].split()))
        idle_end = list(map(float, idle_end[0].split()))

        # Insert the start time of the vehicle to the first idle_start and the end time of the vehicle to the last idle_end
        idle_start.insert(0, vehicle_start_end[vehicle]['start'])
        idle_end.insert(len(idle_end), vehicle_start_end[vehicle]['end'])

        # Modify the vecvalue column to contain the new values as string in the format 'float float float ... float'
        new_idle_start = ' '.join(map(str, idle_start))
        new_idle_end = ' '.join(map(str, idle_end))

        # Update the vecvalue column with the new values
        df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleStart'), 'vecvalue'] = new_idle_start
        df.loc[(df['vehicle'] == vehicle) & (df['name'] == 'idleEnd'), 'vecvalue'] = new_idle_end

        processed += 1

        # Print the percentage of vehicles processed
        print(f"Second Filter - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')

    # Save to csv
    df.to_csv('second_filtered_results.csv', index=False)
    
    print("Second Filter - Done")

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
    df_original.to_csv('no_disparity_filtered_results.csv', index=False)

    return df_original

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
    df.to_csv('separated_rx_idle_results.csv', index=False)

    print("Separated rx and idle")
    return df

def generate_windows(df, window_size_in_seconds=WINDOW_SIZE):

    window_size_in_ms = window_size_in_seconds * 1000.0

    df = df.copy()

    df = df[df['vehicle'] != 'vehicle']

    # For each vehicle get the rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd values
    grouped_df = df.groupby('vehicle')

    # Get the number of vehicles
    vehicles = df['vehicle'].unique()
    processed = 0

    # Create a list to store the windows for each vehicle
    all_windows = []

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

        # Combine _start data into a single list of tuples (mode, time)
        start = [(mode, time*1000) for mode, time in zip(['rx']*len(rx_start) + ['idle']*len(idle_start) + ['tx']*len(tx_start), rx_start + idle_start + tx_start)]

        # Sort data by time and save the new order to apply to _end data
        index_order = np.argsort([time for mode, time in start])
        start = [start[i] for i in index_order]

        # Combine _end data into a single list of tuples (mode, time)
        end = [(mode, time*1000) for mode, time in zip(['rx']*len(rx_end) + ['idle']*len(idle_end) + ['tx']*len(tx_end), rx_end + idle_end + tx_end)]

        # Sort data by time using the order from the _start data
        end = [end[i] for i in index_order]

        # Convert sorted data back to NumPy arrays
        start = np.array(start)
        end = np.array(end)

        np.printoptions(precision=20)
        # Calculate durations by subtracting the second column of end from the second column of start but keep the mode
        
        # Durations is the same shape as start and end
        durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

        # Check if there is any negative 
        """ try:
            if np.any(durations[:, 1].astype(float) < 0.0):
                print(f"Negative duration for vehicle {vehicle}")
        except:
            print(f"Negative duration for vehicle {vehicle}") """


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

                duration_index += 1

            else:
                # If the windows list has the index window_index + 1, set the current_duration to the duration of the first element of the next window
                try:
                    current_duration = float(windows_for_vehicle[window_index + 1][0][1])
                except:
                    current_duration = 0
                
                # Check if the window duration is equal to the window size
                sum_duration = 0
                for duration in windows_for_vehicle[window_index]:
                    sum_duration += float(duration[1])

                if sum_duration != window_size_in_ms:
                    print(f"Vehicle {vehicle} - Window {window_index} - Sum duration: {sum_duration}")


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

        # Count the total number of durations within the windows
        """ total_durations = 0
        for window in windows_for_vehicle:
            total_durations += len(window)
        
        print(f"Total number of durations: {total_durations}")

        # Print the last duration from the last window
        print(f"Last duration from the last window: {windows_for_vehicle[-1][-5:]}")
        # Print the last duration from the durations
        print(f"Last duration from the durations: {durations[-5:]}") """


        # Append the windows for the vehicle to the all_windows list
        all_windows.append(windows_for_vehicle)

        processed += 1
        print(f"Generate Windows - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')    
    
    
    print("Number of vehicles in all_windows", len(all_windows))
    total_number_of_windows = 0

    print("Saving to hdf5 file (this takes a while)")
    # Remove the previous file if it exists
    try:
        os.remove(f'windows{window_size_in_seconds}s.hdf5')
    except:
        pass
    # Save to hdf5 file
    with h5py.File(f'windows{window_size_in_seconds}s.hdf5', 'a') as file:
        for i, vehicle_windows in enumerate(all_windows):
            for j, window in enumerate(vehicle_windows):
                total_number_of_windows += 1

                # Change the type to S32
                window = np.array(window, dtype='S32')
                
                file.create_dataset(f'vehicle_{i}/window_{j}', data=window)

    print(f"Total number of windows: {total_number_of_windows}")

    print("Generated windows")

def generate_X_and_Y():

    print("Generating X and Y data")

    with h5py.File(f'Data/windows{WINDOW_SIZE}s.hdf5', 'r') as file:
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

    # Generate X and Y data
    # Where X are windows and Y are the immediate following windows
    X = []
    Y = []
    for vehicle_key, vehicle_data in all_data.items():
        for i in range(len(vehicle_data) - 1):
            X.append(vehicle_data[i])
            Y.append(vehicle_data[i+1])

    # For each window, both in X and Y, summarize it to have only 3 values: Total duration in idle, total duration in rx and total duration in tx
    # Considering that each window has multiple lists in the format [mode, duration]
    windows_to_remove = []
    processed_windows = 0
    for i in range(len(X)):

        # X
        idle = 0
        rx = 0
        tx = 0
        for j in range(len(X[i])):
            if X[i][j][0] == 'idle':
                idle += float(X[i][j][1])
            elif X[i][j][0] == 'rx':
                rx += float(X[i][j][1])
            elif X[i][j][0] == 'tx':
                tx += float(X[i][j][1])
        
        X[i] = [idle, rx, tx]

        # Y
        idle = 0
        rx = 0
        tx = 0
        for j in range(len(Y[i])):
            if Y[i][j][0] == 'idle':
                idle += float(Y[i][j][1])
            elif Y[i][j][0] == 'rx':
                rx += float(Y[i][j][1])
            elif Y[i][j][0] == 'tx':
                tx += float(Y[i][j][1])

        Y[i] = [idle, rx, tx]

        # If the sum of idle, rx and tx is not close to window_size_in_seconds*1000, then mark the window to be removed
        if abs(sum(X[i]) - WINDOW_SIZE*1000) > WINDOW_SIZE*1000*0.02 or abs(sum(Y[i]) - WINDOW_SIZE*1000) > WINDOW_SIZE*1000*0.02:
            """ print(f"Window {i} is not equal to {window_size_in_seconds}s")
            print(f"X: {X[i]}")
            print(f"Y: {Y[i]}")
            print(f"Sum X: {sum(X[i])}")
            print(f"Sum Y: {sum(Y[i])}") """
            windows_to_remove.append(i)

        # Print the percentage of windows processed
        processed_windows += 1
        print(f"Generate X and Y - Processed {processed_windows/len(X)*100:.2f}% of windows", end='\r')


    # Print the number of windows to be removed
    print(f"Number of windows to be removed: {len(windows_to_remove)}")

    # Remove the windows that are not equal to window_size_in_seconds
    # Sort the indices in descending order
    windows_to_remove_sorted = sorted(windows_to_remove, reverse=True)

    # Remove the elements from the lists
    for i in windows_to_remove_sorted:
        X.pop(i)
        Y.pop(i)

    # Save to a npy file
    np.save('Data/X.npy', X)
    np.save('Data/Y.npy', Y)

if __name__ == "__main__":

    # Track memory usage before loading data
    process = psutil.Process()
    mem_before = process.memory_info().rss  # Resident Set Size (RAM usage)

    # This script generates windows of the data from the raw simulation data. Uncomment all to do all steps.
    # You can comment steps already done to avoid repeating them.

    """ # Apply first filter to the raw data
    raw_file_path = 'results.tar.xz'
    df = pd.read_csv(raw_file_path, compression='infer', chunksize=10000)
    first_filter(df)

    # Apply second filter to the first filtered data
    first_filtered_path = "first_filtered_results.csv"
    df = pd.read_csv(first_filtered_path)
    df = reduce_mem_usage(df)
    second_filter(df)

    # Test disparities
    second_filtered_path = "second_filtered_results.csv"
    df = reduce_mem_usage(df)
    df = pd.read_csv(second_filtered_path)
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
    no_disparity_filtered_path = "no_disparity_filtered_results.csv"
    df = pd.read_csv(no_disparity_filtered_path)
    df = reduce_mem_usage(df)
    df = separate_rx_idle(df) 
    
   
    # Generate windows
    separated_rx_idle_path = "separated_rx_idle_results.csv"
    df = pd.read_csv(separated_rx_idle_path)
    df = reduce_mem_usage(df)
    generate_windows(df)
   
    # Generate X and Y data
    generate_X_and_Y()
    

    # Track memory usage after loading data
    mem_after = process.memory_info().rss

    # Calculate memory used
    memory_used = (mem_after - mem_before) / (1024 * 1024)  # Convert to MB

    # Print memory usage after processing
    print(f"Memory used (GB): {memory_used/1024:.2f}")

    """