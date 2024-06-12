import matplotlib.pyplot as plt

def plot_results(Y_test, predictions, model_name, mae_all, mae_test, past_windows):
        plt.figure(figsize=(15, 15))
        for i in range(3):
            plt.subplot(3, 1, i+1)
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