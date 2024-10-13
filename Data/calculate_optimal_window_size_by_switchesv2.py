from itertools import permutations
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gc
import multiprocessing as mp
import pyarrow.parquet as pq
import pyarrow as pa
import sys
import tracemalloc

from paretoset import paretoset

DATA_PATH = 'Data/iot/' # Leave the last slash

# ---------------- HELPER FUNCTIONS FOR THE CODES THAT CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES ------------------------

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

""" def find_optimal_window_sizes(variance_scores, window_sizes):
    min_value = np.min(variance_scores)
    min_indices = np.where(variance_scores == min_value)[0] # Instead of [0]
    optimal_window_sizes = [window_sizes[idx] for idx in min_indices]

    # Get every window size with less than 0.3 CV
    
    indices = np.where(np.array(variance_scores) <= 0.3)[0]
    optimal_window_sizes_less_than_02 = [window_sizes[idx] for idx in indices]
    print(f"\nCV:{min_value} for OPTIMAL WINDOW SIZES LESS THAN 0.3: \n{optimal_window_sizes_less_than_02}\n")

    print(f"\nCV:{min_value} for OPTIMAL WINDOW SIZES: \n{optimal_window_sizes}\n")
    #return [min(optimal_window_sizes)] if optimal_window_sizes else []
    return optimal_window_sizes_less_than_02 """

def find_optimal_window_sizes(variance_scores, window_sizes):
    
    optimization_points = pd.DataFrame({'window_size': window_sizes, 'cv_score': variance_scores})
    mask = paretoset(optimization_points, sense=['max', 'min'])
    points = optimization_points[mask]

    # Get the optimal window sizes
    pareto_window_sizes = points['window_size'].values

    print(f"\nCV:{points['cv_score'].values} for OPTIMAL WINDOW SIZES: \n{pareto_window_sizes}\n")
    
    return pareto_window_sizes

# ----------------- END HELPER FUNCTIONS -----------------------------------------

# ---------------------------- PARALLEL CODE TO CALCULATE THE OPTIMAL WINDOW SIZE BY SWITCHES --------------------------------------------------

def sturges(n: int):
    result = np.ceil(np.log2(n))
    return int(result) + 1

def process_vehicle(group, window_sizes):
    #print(f"Processing vehicle: {group['vehicle'].values[0]}", flush=True)
    start_time = time.time()
    # Get the vecvalues for rxStart, rxEnd, idleStart, idleEnd, txStart, and txEnd if they exist
    rx_start = group.loc[group['name'] == 'rxStart', 'vecvalue'].values
    rx_end = group.loc[group['name'] == 'rxEnd', 'vecvalue'].values
    idle_start = group.loc[group['name'] == 'idleStart', 'vecvalue'].values
    idle_end = group.loc[group['name'] == 'idleEnd', 'vecvalue'].values
    tx_start = group.loc[group['name'] == 'txStart', 'vecvalue'].values
    tx_end = group.loc[group['name'] == 'txEnd', 'vecvalue'].values

    switching_start = group.loc[group['name'] == 'switchingStart', 'vecvalue'].values
    switching_end = group.loc[group['name'] == 'switchingEnd', 'vecvalue'].values
    
    # Convert them to list of float by splitting the string if they exist
    rx_start = list(map(float, rx_start[0].split())) if len(rx_start) > 0 else []
    rx_end = list(map(float, rx_end[0].split())) if len(rx_end) > 0 else []
    idle_start = list(map(float, idle_start[0].split())) if len(idle_start) > 0 else []
    idle_end = list(map(float, idle_end[0].split())) if len(idle_end) > 0 else []
    tx_start = list(map(float, tx_start[0].split())) if len(tx_start) > 0 else []
    tx_end = list(map(float, tx_end[0].split())) if len(tx_end) > 0 else []

    switching_start = list(map(float, switching_start[0].split())) if len(switching_start) > 0 else []
    switching_end = list(map(float, switching_end[0].split())) if len(switching_end) > 0 else []

    #print(f"Vehicle: {group['vehicle'].values[0]} - RX Start: {len(rx_start)} - RX End: {len(rx_end)} - Idle Start: {len(idle_start)} - Idle End: {len(idle_end)} - TX Start: {len(tx_start)} - TX End: {len(tx_end)} - Switching Start: {len(switching_start)} - Switching End: {len(switching_end)}", flush=True)
    #print(f"First RX Start: {rx_start[0]} - First RX End: {rx_end[0]} - First Idle Start: {idle_start[0]} - First Idle End: {idle_end[0]} - First TX Start: {tx_start[0]} - First TX End: {tx_end[0]} - First Switching Start: {switching_start[0]} - First Switching End: {switching_end[0]}", flush=True)
    #print(f"Last RX Start: {rx_start[-1]} - Last RX End: {rx_end[-1]} - Last Idle Start: {idle_start[-1]} - Last Idle End: {idle_end[-1]} - Last TX Start: {tx_start[-1]} - Last TX End: {tx_end[-1]} - Last Switching Start: {switching_start[-1]} - Last Switching End: {switching_end[-1]}", flush=True)
          
    
    # Combine _start data into a single list of tuples (mode, time)
    start = [(mode, time * 1000) for mode, time in zip(['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start), 
                                rx_start + idle_start + tx_start)]

    # Sort data by time and save the new order to apply to _end data
    index_order = np.argsort([time for mode, time in start])
    start = [start[i] for i in index_order]

    # Combine _end data into a single list of tuples (mode, time)
    end = [(mode, time * 1000) for mode, time in zip(['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end), 
                                rx_end + idle_end + tx_end)]

    # Sort data by time using the order from the _start data
    end = [end[i] for i in index_order]

    # Convert sorted data back to NumPy arrays
    start = np.array(start)
    end = np.array(end)

    # Calculate durations by subtracting the second column of end from the second column of start but keep the mode
    durations = np.column_stack((start[:, 0], end[:, 1].astype(float) - start[:, 1].astype(float)))

    #print(f"Vehicle: {group['vehicle'].values[0]} - DURATIONS: {durations}", flush=True)

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

# ---------------------------- END OF THE PARALELL CODE TO CALCULATE OPTIMAL WINDOW SIZE BY SWITCHES --------------------

def process_large_parquet(filepath, window_sizes, batch_size=100):
    parquet_file = pq.ParquetFile(filepath)
    incomplete_vehicles = {}  # To store vehicles split across batches
    mean_time_intervals = []
    processed = 0
    
    # Initialize multiprocessing Pool once
    pool = mp.Pool(processes=24)
    
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()
        
            vehicle_groups = batch_df.groupby('vehicle')
            
            async_results = []
            
            for vehicle, group in vehicle_groups:
                if vehicle in incomplete_vehicles:
                    # Append to existing incomplete vehicle data
                    incomplete_vehicles[vehicle] = pd.concat([incomplete_vehicles[vehicle], group], ignore_index=True)
                else:
                    incomplete_vehicles[vehicle] = group
                
                # If we have all 8 lines for the vehicle, submit it for processing
                if len(incomplete_vehicles[vehicle]) >= 8:
                    vehicle_data = incomplete_vehicles.pop(vehicle).head(8)
                    # Submit the vehicle_data to the pool
                    async_result = pool.apply_async(process_vehicle, args=(vehicle_data, window_sizes))
                    async_results.append(async_result)
            
            # Collect results from async tasks
            for async_result in async_results:
                try:
                    result = async_result.get()  # Adjust timeout as needed
                    mean_time_intervals.extend(result)
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"Processed {processed} vehicles", flush=True)
                except mp.TimeoutError:
                    print("A task timed out.", flush=True)
                except Exception as e:
                    print(f"Error processing a vehicle: {e}", flush=True)
            
            # Clean up
            del batch_df, vehicle_groups, async_results
            gc.collect()
            #if processed >= 100:
            #    break # Remove this line to process all batches
        
        # After all batches are processed, handle remaining incomplete vehicles
        for vehicle, vehicle_data in incomplete_vehicles.items():
            if len(vehicle_data) >= 8:
                vehicle_data = vehicle_data.head(8)
                result = pool.apply_async(process_vehicle, args=(vehicle_data, window_sizes))
                try:
                    mean_time_interval_ms = result.get(timeout=1)
                    mean_time_intervals.extend(mean_time_interval_ms)
                    processed += 1
                except Exception as e:
                    print(f"Error processing a vehicle: {e}", flush=True)
        
        # Finalize Pool
        pool.close()
        pool.join()
        
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        pool.terminate()
        pool.join()
        raise e
    
    # Calculate statistics
    mean_time_intervals = np.nan_to_num(mean_time_intervals)
    mean_mean_time_interval_ms = np.mean(mean_time_intervals)
    variance_time_interval_ms = np.std(mean_time_intervals)
    
    print(f"\nMean time interval across all vehicles for mean optimal window size in milliseconds: {mean_mean_time_interval_ms:.2f}")
    print(f"Variance time interval across all vehicles for mean optimal window size in milliseconds: {variance_time_interval_ms}")
    
    # Plot histogram
    mean_value = np.mean(mean_time_intervals)
    median_value = np.median(mean_time_intervals)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_time_intervals, bins=sturges(len(mean_time_intervals)), color='skyblue', edgecolor='black')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.title('Histogram of Mean Time Intervals', fontsize=18)
    plt.xlabel('Mean Time Interval (ms)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(f'{DATA_PATH}histogram_mean_optimal_time_window_by_switches.pdf')
    plt.close()
    
    return mean_mean_time_interval_ms

def mem_usage():
    snapshot = tracemalloc.take_snapshot() 
    top_stats = snapshot.statistics('lineno') 
    
    for stat in top_stats[:2]: 
        print(stat, flush=True)

if __name__ == "__main__":
    # Calculate the optimal window size
    separated_rx_idle_path = f"{DATA_PATH}combined.parquet"
    window_sizes = list(range(30, 1100, 30))
    optimal_window_size_in_ms = process_large_parquet(separated_rx_idle_path, window_sizes)