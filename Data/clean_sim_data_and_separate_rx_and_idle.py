import psutil
import gc
import numpy as np
import pandas as pd 
import json
import csv

WINDOW_SIZE = 0.245 # PUT THE WINDOW SIZE HERE IN SECONDS
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

if __name__ == "__main__":

    # This script preprocesses the simulation data. Uncomment all to do all steps.
    # You can comment steps already done to avoid repeating them.

    # Apply first filter to the raw data
    #raw_file_path = f'{DATA_PATH}results.tar.xz'
    #df = pd.read_csv(raw_file_path, compression='infer', chunksize=10000)
    raw_file_path = f'{DATA_PATH}results_cam_sizes.csv' # PUT YOUR INITIAL CSV FILE HERE.
    df = pd.read_csv(raw_file_path, na_values=['', 'NA'], keep_default_na=True, on_bad_lines='skip', quoting=csv.QUOTE_NONE)
    first_filter(df)

    # Apply second filter to the first filtered data
    first_filtered_path = f"{DATA_PATH}first_filtered_results.csv"
    df = pd.read_csv(first_filtered_path)
    df = reduce_mem_usage(df)
    second_filter(df)

    # Test disparities
    second_filtered_path = f"{DATA_PATH}second_filtered_results.csv"
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
