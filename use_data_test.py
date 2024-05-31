import h5py
import psutil
import numpy as np

if __name__ == "__main__":
    window_size_in_seconds = 10
    # Track memory usage before loading data
    process = psutil.Process()
    mem_before = process.memory_info().rss  # Resident Set Size (RAM usage)

    # Open the HDF5 file in read mode
    with h5py.File(f'Data/windows{window_size_in_seconds}s.hdf5', 'r') as file:
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

    # Now `all_data` contains all the retrieved data
    
    # Track memory usage after loading data
    mem_after = process.memory_info().rss

    # Calculate memory used
    memory_used = (mem_after - mem_before) / (1024 * 1024)  # Convert to MB

    # Print the number of vehicles
    print(f"Number of vehicles: {len(all_data)}")

    total_windows = 0
    # For each vehicle, print each window length
    for vehicle_key, vehicle_data in all_data.items():
        for window_data in vehicle_data:
            total_windows += 1
            #print(f"Vehicle {vehicle_key}: Window length: {len(window_data)}")

    # Print the total number of windows
    print(f"Total number of windows: {total_windows}")

    # Print memory usage after processing
    print(f"Memory used (GB): {memory_used/1024:.2f}")

    # Generate X and Y data
    # Where X are windows and Y are the immediate following windows
    X = []
    Y = []
    for vehicle_key, vehicle_data in all_data.items():
        for i in range(len(vehicle_data) - 1):
            X.append(vehicle_data[i])
            Y.append(vehicle_data[i+1])

    # Print the length of X and Y
    print(f"Length of X: {len(X)}")
    print(f"Length of Y: {len(Y)}")

    # Get the max length of the windows
    max_length = max([len(window) for window in X])
    print(f"Max length of windows: {max_length}")

    max_length = max([len(window) for window in Y])
    print(f"Max length of windows: {max_length}")

    # Get the min length of the windows
    min_length = min([len(window) for window in X])
    print(f"Min length of windows: {min_length}")

    min_length = min([len(window) for window in Y])
    print(f"Min length of windows: {min_length}")

    # For each window, both in X and Y, summarize it to have only 3 values: Total duration in idle, total duration in rx and total duration in tx
    # Considering that each window has multiple lists in the format [mode, duration]
    windows_to_remove = []

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
        if abs(sum(X[i]) - window_size_in_seconds*1000) > window_size_in_seconds*1000*0.02 or abs(sum(Y[i]) - window_size_in_seconds*1000) > window_size_in_seconds*1000*0.02:
            """ print(f"Window {i} is not equal to {window_size_in_seconds}s")
            print(f"X: {X[i]}")
            print(f"Y: {Y[i]}")
            print(f"Sum X: {sum(X[i])}")
            print(f"Sum Y: {sum(Y[i])}") """
            windows_to_remove.append(i)

    # Print the number of windows to be removed
    print(f"Number of windows to be removed: {len(windows_to_remove)}")

    # Remove the windows that are not equal to window_size_in_seconds
    # Sort the indices in descending order
    windows_to_remove_sorted = sorted(windows_to_remove, reverse=True)

    # Remove the elements from the lists
    for i in windows_to_remove_sorted:
        X.pop(i)
        Y.pop(i)


    # Print the first 5 elements of X and Y
    print("X:")
    for i in range(50, 55):
        print(X[i])
    print("Y:")
    for i in range(50, 55):
        print(Y[i])