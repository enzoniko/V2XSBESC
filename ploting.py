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

# Function to plot error histograms for multiple models
# Function to plot error histograms for multiple models
def plot_error_histograms(models_results, past_windows=1):
    features = ['IDLE', 'RX']
    
    plt.figure(figsize=(8, 10))

    for i, feature in enumerate(features):
        plt.subplot(2, 1, i + 1)
        for model_name, results in models_results.items():
            Y_test, predictions, mae_all, mae_test, mae_test_std = results
            
            # Calculate errors
            errors = Y_test - predictions
            plt.hist(errors[:, i], bins=sturges(len(errors)), alpha=0.7, edgecolor='black', label=model_name, histtype='stepfilled')

        plt.title(f'Error Histogram for Feature {feature}', fontsize=18)
        plt.xlabel('Error (ms)', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)

    # Add a main title for the figure
    plt.suptitle("Error Histograms for Multiple Models", fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title

    # Save the plot
    plt.savefig(f"Images/error_histograms_with_{past_windows}_past_windows.png")
    plt.close()

def plot_error_vs_past_windows(results):
    plt.figure(figsize=(10, 6))  # Adjust figure size for better presentation
    markers = ['o', 's', '^', 'D', 'x', 'P']  # Marker styles for each model (add more if needed)
    
    # Change the ANN name to MLP
    results['MLP'] = results.pop('ANN')

    # Ensure the order of models is Random Forest, MLP, XGBoost, and Moving Average
    models_order = ['Random Forest', 'MLP', 'XGBoost', 'Moving Average']

    results = {model_name: results[model_name] for model_name in models_order}

    for idx, (model_name, maes) in enumerate(results.items()):
        past_windows = range(1, len(maes) + 1)
        plt.plot(past_windows, maes, marker=markers[idx], markersize=8, label=model_name, linewidth=2)
    
    #plt.title("Error vs Past Windows", fontsize=18)
    plt.xlabel("Past Windows", fontsize=18)
    plt.ylabel("Mean Absolute Error", fontsize=18)
    plt.xticks(range(1, len(maes) + 1), fontsize=18)  # Adjust X-axis ticks
    plt.yticks(fontsize=18)  # Adjust Y-axis ticks
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()  # Ensure tight layout for better spacing
    plt.savefig("Images/error_vs_past_windows.pdf")  # Save with higher resolution
    #plt.show()
    plt.close()