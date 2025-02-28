def calculateAveragemAhperInference(mAh, experimentTime, inferenceTime):
     # Hardcoded average Baseline power consumption in mAh
    baseline = 68

    # Subtract the baseline power consumption from the total power consumption
    mAh = mAh - baseline
    
    # Transform inferenceTime from microseconds to seconds
    inferenceTime = inferenceTime / 1000000
    
    # Calculate the number of inferences in the experiment
    inferences = (experimentTime * 60)/inferenceTime

    # Calculate the average mAh per inference
    average_mAh_per_inference = mAh/inferences

    print(f"Average mAh per inference: {average_mAh_per_inference}")

    # Call the function to convert mAh to Joules and pass the average mAh per inference and the average voltage
    mAhtoJoules(average_mAh_per_inference, 5.26)

def mAhtoJoules(mAh, voltage):

    # Calculate the energy in Joules
    energy = mAh * voltage * 3.6
    print(f"Energy in Joules: {energy}")

if __name__ == '__main__':

    # Average baseline total power consumption in mAh during 5 minutes:
    # 44.375 mAh
    # Average voltage in V:
    # 5.18 V

    # Average total power consumption in mAh for MLP (ANN) model:
    # 72 mAh 
    # Average inference time in microseconds for MLP (ANN) model:
    # 0.58 microseconds +- 0.009 microseconds
    # Experiment time in minutes: 5 minutes -> 300 seconds

    # Average total power consumption in mAh for RF model:
    # 73 mAh
    # Average inference time in microseconds for RF model:
    # 1.58 microseconds +- 0.02 microseconds
    # Experiment time in minutes: 5 minutes -> 300 seconds

    # Average total power consumption in mAh for XGB model:
    # 82.5 mAh
    # Average inference time in microseconds for XGB model:
    # 3.44 microseconds +- 0.103 microseconds
    # Experiment time in minutes: 5 minutes -> 300 seconds

    # Call the function to calculate the average mAh per inference and pass the average total power consumption in mAh, the experiment time in minutes and the average inference time in microseconds
    # We could improve this by adding the standard deviation of the inference time as a parameter 
    # This would allow us to calculate the average mAh per inference with a confidence interval


    # FOR MLP MODEL
    print("MLP MODEL:", end=" ")
    calculateAveragemAhperInference(84, 5, 2.82)
    print()

    # FOR RF MODEL
    print("RF MODEL:", end=" ")
    calculateAveragemAhperInference(85, 5, 18.54)
    print()

    # FOR XGB MODEL
    print("XGB MODEL:", end=" ")
    calculateAveragemAhperInference(93, 5, 3.44)