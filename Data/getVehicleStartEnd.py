import json
DATA_PATH = 'Data/data_with_noise/'
import matplotlib.pyplot as plt
if __name__ == "__main__":

    # Open the file "General-#0.sca"
    with open(f"{DATA_PATH}General-#0.sca", "r") as file:
        lines = file.readlines()

    # Get the lines that start with "scalar"
    scalar_lines = [line for line in lines if line.startswith("scalar")]

    # Get the lines that have "startTime" or "stopTime" in them
    start_lines = [line for line in scalar_lines if "startTime" in line]
    end_lines = [line for line in scalar_lines if "stopTime" in line]

    # Create a dictionary to store the start and end times for each vehicle
    vehicle_start_end = {}

    # For each start line, get the vehicle number and the start time
    for line in start_lines:
        # The vehicle number is within '[' and ']' in the line
        vehicle = int(line[line.find("[")+1:line.find("]")])

        # The start time is the last value in the line
        start_time = float(line.split()[-1])

        # Store the start time in the dictionary
        vehicle_start_end[vehicle] = {"start": start_time, "end": None}

    # For each end line, get the vehicle number and the end time
    for line in end_lines:
        # The vehicle number is within '[' and ']' in the line
        vehicle = int(line[line.find("[")+1:line.find("]")])

        # The end time is the last value in the line
        end_time = float(line.split()[-1])

        # Store the end time in the dictionary
        vehicle_start_end[vehicle]["end"] = end_time
    
    vehicles_less_10s = [vehicle for vehicle, times in vehicle_start_end.items() if times["end"] - times["start"] < 28]
    print(f"Vehicles that took less than 28 seconds to complete the simulation: {[vehicle for vehicle in vehicles_less_10s]}")
    print(f"Count: {len(vehicles_less_10s)}")

    # Get the average time for all vehicles and the standard deviation
    times = [times["end"] - times["start"] for times in vehicle_start_end.values()]
    average_time = sum(times) / len(times)
    std_dev = (sum((time - average_time) ** 2 for time in times) / len(times)) ** 0.5
    print(f"Average time for all vehicles: {average_time}")
    print(f"Standard deviation: {std_dev}")
    
    # Plot a histogram of the times
    plt.hist(times, bins=20)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Simulation Times")
    plt.show()

    # Print the number of vehicles
    print(f"Number of vehicles: {len(vehicle_start_end)}")

    
    # Print the start and end times for each vehicle
    """ for vehicle, times in vehicle_start_end.items():
        print(f"Vehicle {vehicle}: Start time: {times['start']}, End time: {times['end']}") """

    # Save the vehicle start and end times to a file
    with open(f"{DATA_PATH}vehicle_start_end.json", "w") as file:
        json.dump(vehicle_start_end, file, indent=4)