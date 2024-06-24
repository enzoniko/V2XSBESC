from itertools import permutations
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gc
import multiprocessing as mp

DATA_PATH = 'Data/data_with_noise/' # Leave the last slash

# ---------------- HELPER FUNCTIONS FOR THE CODES THAT CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES ------------------------

""" def count_mode_switches(df, window_size):
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

    return switch_counts """

def count_mode_switches(df, window_size):
    switch_counts = []
    modes = df['mode'].values
    for i in range(0, len(modes) - window_size + 1, window_size):
        window = modes[i:i + window_size]
        window_switch_counts = {}
        for mode1, mode2 in permutations(['rx', 'tx', 'idle'], 2):
            count = sum((window[j] == mode1 and window[j + 1] == mode2) or
                        (window[j] == mode2 and window[j + 1] == mode1)
                        for j in range(len(window) - 1))
            window_switch_counts[(mode1, mode2)] = count

        window_switch_counts = {k: v for k, v in window_switch_counts.items() if v > 0}

        switch_counts.append(window_switch_counts)

    #print('--------------------------')
    #print(f'\n SWITCH COUNTS: {switch_counts}\n', flush=True)
    #print('--------------------------')

    return switch_counts

""" def evaluate_window_sizes(df, window_sizes):
    variance_scores = []
    processed = 0
    for window_size in window_sizes:
        switch_counts = count_mode_switches(df, window_size)
        switch_counts_flat = [switch_count for window_switch_counts in switch_counts for switch_count in window_switch_counts.values()]
        variance_score = np.var(switch_counts_flat)
        variance_scores.append(variance_score)
        processed += 1
        #print(f'Evaluated {processed*100/len(window_sizes)}% window sizes for the vehicle', end='\r', flush=True)
    
    print(variance_scores, flush=True)
    return variance_scores """

""" def evaluate_window_sizes(df, window_sizes):
    variance_scores = []
    processed = 0
    for window_size in window_sizes:
        switch_counts = count_mode_switches(df, window_size)
        
        # Aggregate switch counts for each mode combination
        all_switch_counts = {}
        for window_switch_counts in switch_counts:
            for key, count in window_switch_counts.items():
                if key not in all_switch_counts:
                    all_switch_counts[key] = []
                all_switch_counts[key].append(count)
        
        # print(f"\nAGGREGATED SWITCH COUNTS: \n{all_switch_counts}\n")

        # Calculate variance for each mode combination
        variances = []
        for counts in all_switch_counts.values():
            variances.append(np.var(counts))
        
        # Calculate the average of these variances
        average_variance = np.mean(variances)
        variance_scores.append(average_variance)
        
        processed += 1
        #print(f'Evaluated {processed*100/len(window_sizes):.2f}% window sizes for the vehicle', end='\r', flush=True)
    
    print('--------------------------')
    print(f'\nVARIANCE SCORES: {variance_scores}\n', flush=True)
    print('--------------------------')
    return variance_scores """

def evaluate_window_sizes(df, window_sizes):
    cv_scores = []
    for window_size in window_sizes:
        switch_counts = count_mode_switches(df, window_size)
        
        # Aggregate switch counts for each mode combination
        all_switch_counts = {}
        for window_switch_counts in switch_counts:
            for key, count in window_switch_counts.items():
                if key not in all_switch_counts:
                    all_switch_counts[key] = []
                all_switch_counts[key].append(count)
        
        # Calculate coefficient of variation (CV) for each mode combination
        cvs = []
        for counts in all_switch_counts.values():
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            cv = std_count / mean_count
            cvs.append(cv)
        
        # Calculate the average of these coefficients of variation
        average_cv = np.mean(cvs)
        cv_scores.append(average_cv)
        
    #print(f'Evaluated window sizes, Average CVs: {cv_scores}')
    
    return cv_scores

""" def find_optimal_window_size(variance_scores, window_sizes):
    best_window_size = window_sizes[np.argmin(variance_scores)]
    print(f"BEST VARIANCE AVERAGE: {np.min(variance_scores)}")
    return best_window_size """

def find_optimal_window_sizes(variance_scores, window_sizes):
    min_value = np.min(variance_scores)
    min_indices = np.where(variance_scores == min_value)[0]
    optimal_window_sizes = [window_sizes[idx] for idx in min_indices]



    print(f"\nOPTIMAL WINDOW SIZES: \n{optimal_window_sizes}\n")
    return [min(optimal_window_sizes)] if optimal_window_sizes else []

# ----------------- END HELPER FUNCTIONS -----------------------------------------

# ---------------------------- PARALLEL CODE TO CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES --------------------------------------------------

def sturges(n: int):
    result = np.ceil(np.log2(n))
    return int(result) + 1

""" def process_vehicle(group, window_sizes):
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

    return mean_time_interval_ms """

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

    # Evaluate window sizes and find optimal window sizes
    variance_scores = evaluate_window_sizes(durations_df, window_sizes)
    optimal_window_sizes = find_optimal_window_sizes(variance_scores, window_sizes)

    # Calculate the average time interval for each optimal window size
    mean_time_intervals = []
    for best_window_size in optimal_window_sizes:
        optimal_window_durations = durations_df['duration'].rolling(window=best_window_size).sum().dropna()
        mean_time_interval_ms = optimal_window_durations.mean()
        mean_time_intervals.append(mean_time_interval_ms)

    print(f"Time taken for vehicle: {time.time() - start_time}", flush=True)

    return mean_time_intervals

""" def calculate_optimal_window_size_by_switches(df) -> int:

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
    with mp.Pool(processes=1) as pool:
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

    return mean_mean_time_interval_ms """

def calculate_optimal_window_size_by_switches(df) -> float:

    df = df.copy()

    df = df[df['vehicle'] != 'vehicle']

    vehicles = df['vehicle'].unique()

    downsample_factor = 1
    # Get only two vehicles to debug
    df = df[df['vehicle'].isin(random.sample(range(len(vehicles)), len(vehicles)//downsample_factor))]
    grouped_df = df.groupby('vehicle')

    processed = 0

    # Use multiprocessing Pool to parallelize vehicle processing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [pool.apply_async(process_vehicle, args=(group, range(30, 2001, 10))) for _, group in grouped_df]

        mean_time_intervals = []
        for result in results:
            mean_time_interval_ms = result.get()
            mean_time_intervals.extend(mean_time_interval_ms)
            processed += 1
            print(f"Calculating Optimal Window Size - Processed {processed/len(vehicles)*100:.2f}% of vehicles", end='\r')

    # Calculate the mean time interval across all vehicles for the mean optimal window size
    mean_mean_time_interval_ms = np.mean(mean_time_intervals)
    print(f"Mean time interval across all vehicles for mean optimal window size in milliseconds: {mean_mean_time_interval_ms:.2f}")

    print(f"Variance time interval across all vehicles for mean optimal window size in milliseconds: {np.std(mean_time_intervals)}")

    mean_value = np.mean(mean_time_intervals)
    median_value = np.median(mean_time_intervals)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_time_intervals, bins='sturges', color='skyblue', edgecolor='black')
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

    # Calculate the optimal window size
    separated_rx_idle_path = f"{DATA_PATH}separated_rx_idle_results.csv"
    df = pd.read_csv(separated_rx_idle_path)
    df = reduce_mem_usage(df)
    #optimal_window_size_in_ms = calculate_optimal_window_size(df)
    optimal_window_size_in_ms = calculate_optimal_window_size_by_switches(df)