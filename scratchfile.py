
""" 
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

 """

from preprocessing import preprocess
from models import BaselineModels
from Data.generate_x_and_y import generate_X_and_Y
from ploting import plot_error_vs_past_windows

def main():

    # All possibilities of number of past_windows to experiment with
    past_windows_possibilities = list(range(1, 11))

    results = {
        "Linear Regression": [],
        "Random Forest": [],
        "Gradient Boosting": [],
        "MLP": []
    }

    for past_windows in past_windows_possibilities:
        generate_X_and_Y(past_windows)

        # Load the dataframes
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = preprocess()

        # Train and evaluate the baseline models
        baseline_models = BaselineModels()

        maes = baseline_models.train_and_evaluate(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler, past_windows)

        for model_name, mae in maes.items():
            results[model_name].append(mae)

        plot_error_vs_past_windows(results)


if __name__ == "__main__":
    main()