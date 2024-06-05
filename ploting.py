import matplotlib.pyplot as plt

def plot_results(Y_test, predictions, model_name, mae, mse):
        plt.figure(figsize=(15, 15))
        for i, column in enumerate(Y_test.columns):
            plt.subplot(3, 1, i+1)
            plt.plot(Y_test[column], label="True")
            plt.plot(predictions[:, i], label="Predicted")
            plt.title(column)
            plt.legend()

        # Adding textboxes for MSE and MAE, centered at the top of the figure
        plt.figtext(0.5, 0.96, f"{model_name} Results", ha="center", fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.92, f"MSE: {mse}", ha="center", fontsize=12)
        plt.figtext(0.5, 0.89, f"MAE: {mae}", ha="center", fontsize=12)
        
        # Add a space between the textboxes and the plots
        plt.subplots_adjust(top=0.85)
        # Save the plot
        plt.savefig(f"Images/{model_name}_results.png")
        plt.close()