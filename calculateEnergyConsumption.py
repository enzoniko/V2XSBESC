if __name__ == '__main__':

    avg_voltage = 5.26

    discharge_capacity = input("Enter the discharge capacity in mAh: ")
    time_taken = input("Enter the time taken in minutes: ")

    # Print power consumption in mW
    mw = (float(discharge_capacity) / (float(time_taken) / 60)) * avg_voltage
    print(f"Power consumption in mW: {mw}")
