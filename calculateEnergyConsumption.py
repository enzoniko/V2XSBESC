def calculateMWfrommAh(mAh):
    avg_voltage = 5.26

    discharge_capacity = input("Enter the discharge capacity in mAh: ")
    time_taken = input("Enter the time taken in minutes: ")

    # Print power consumption in mW
    mw = (float(discharge_capacity) / (float(time_taken) / 60)) * avg_voltage
    print(f"Power consumption in mW: {mw}")

def calculateAveragemAhperInference(mAh, experimentTime, inferenceTime):
    baseline = 0.044375

    # Subtract the baseline power consumption from the total power consumption
    mAh = mAh - baseline
    
    # Transform inferenceTime from microseconds to seconds
    inferenceTime = inferenceTime / 1000000
    
    # Calculate the number of inferences per second
    inferences_per_second = 1 / inferenceTime
    
    inferences = (experimentTime * 60)/inferenceTime
    # Calculate the average mAh per inference
    average_mAh_per_inference = mAh/inferences
    print(f"Average mAh per inference: {average_mAh_per_inference}")
    mAhtoJoules(average_mAh_per_inference, 5.26)
def mAhtoJoules(mAh, voltage):

    # Calculate the energy in Joules
    energy = mAh * voltage * 3.6
    print(f"Energy in Joules: {energy}")
if __name__ == '__main__':

    
    calculateAveragemAhperInference(0.073, 5, 1.58)