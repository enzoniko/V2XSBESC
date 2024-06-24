import matplotlib.pyplot as plt
import numpy as np
""" 
def plot_results(Y_test, predictions, model_name, mae_all, mae_test, past_windows):
        plt.figure(figsize=(15, 15))
        for i in range(2):
            plt.subplot(2, 1, i+1)
            plt.plot(Y_test[:, i], label="True")
            plt.plot(predictions[:, i], label="Predicted")
            plt.title(i)
            plt.legend()

        # Adding textboxes for MAE ALL and MAE TEST, centered at the top of the figure
        plt.figtext(0.5, 0.96, f"{model_name} Results", ha="center", fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.92, f"MAE ALL: {mae_all}", ha="center", fontsize=12)
        plt.figtext(0.5, 0.89, f"MAE TEST: {mae_test}", ha="center", fontsize=12)
        
        # Add a space between the textboxes and the plots
        plt.subplots_adjust(top=0.85)
        # Save the plot
        plt.savefig(f"Images/{model_name}_results_with_{past_windows}_past_windows_.png")
        plt.close()
 """


# CODE TO PLOT PREDICTED VS ACTUAL (GETS POLUTED WITH MANY SAMPLES)

""" def plot_results(Y_test, predictions, model_name, mae_all, mae_test, mae_test_std=1, past_windows=1):
    plt.figure(figsize=(15, 15))
    
    for i in range(2):
        # Sort the values by the true values for the current feature
        sorted_indices = np.argsort(Y_test[:, i])
        Y_test_sorted = Y_test[sorted_indices, i]
        predictions_sorted = predictions[sorted_indices, i]

        plt.subplot(2, 1, i+1)
        plt.plot(Y_test_sorted, label="True", color='blue', marker='o', markersize=2, linestyle='dashed')
        plt.plot(predictions_sorted, label="Predicted", color='red', marker='x', markersize=2, linestyle='dotted')
        plt.title(f'Feature {i}')
        plt.legend()

    # Adding textboxes for MAE ALL and MAE TEST, centered at the top of the figure
    plt.figtext(0.5, 0.96, f"{model_name} Results", ha="center", fontsize=14, fontweight='bold')
    plt.figtext(0.5, 0.92, f"MAE ALL: {mae_all}", ha="center", fontsize=12)
    plt.figtext(0.5, 0.89, f"MAE TEST: {mae_test}", ha="center", fontsize=12)
    plt.figtext(0.5, 0.84, f"MAE TEST STD DEV: {mae_test_std}", ha="center", fontsize=12)
    
    # Add a space between the textboxes and the plots
    plt.subplots_adjust(top=0.80)
    
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"Images/{model_name}_results_with_{past_windows}_past_windows.png")
    plt.close() """

# Sturges helper function
def sturges(n: int):
    result = np.ceil(np.log2(n))
    return int(result) + 1

# SUBSTITUTE CODE TO PLOT A HISTOGRAM INSTEAD, EASIER TO SEE.

def plot_results(Y_test, predictions, model_name, mae_all, mae_test, mae_test_std=1, past_windows=1):
    # Calculate errors
    errors = Y_test - predictions

    features = ['IDLE', 'RX']

    plt.figure(figsize=(5, 10))  # Make the figure more square

    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.hist(errors[:, i], bins=sturges(len(errors)), color='blue', alpha=0.7, edgecolor='black')
        plt.title(f'Error Histogram for Feature {features[i]}', fontsize=14)
        plt.xlabel('Error (ms)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

    # Add text boxes for metrics outside the subplots
    metrics_text = f"MAE ALL: {mae_all}\nMAE TEST: {mae_test}\nMAE TEST STD DEV: {mae_test_std}"
    plt.figtext(0.5, 0.86, metrics_text, ha="center", fontsize=14, bbox={"facecolor": "white", "alpha": 0.7, "pad": 5})

    # Add a main title for the figure
    plt.suptitle(f"{model_name} Error Histogram", fontsize=14, fontweight='bold', y=0.95)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space for the main title and text box

    # Save the plot
    plt.savefig(f"Images/{model_name}_error_histogram_with_{past_windows}_past_windows.png")
    plt.close()

def plot_error_vs_past_windows(results):
    plt.figure(figsize=(15, 15))
    for model_name, maes in results.items():
        plt.plot(maes, label=model_name)
    
    plt.title("Error vs Past Windows")
    plt.xlabel("Past Windows")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.savefig("Images/error_vs_past_windows.png")
    plt.close()