


def experiment_baselineModels_results_vs_past_windows_number():

    from preprocessing import preprocess
    from models import BaselineModels
    from Data.generate_x_and_y import generate_X_and_Y
    from ploting import plot_error_vs_past_windows

    # All possibilities of number of past_windows to experiment with
    past_windows_possibilities = list(range(8, 15))
    past_windows_possibilities = [10]

    results = {
        "Linear Regression": [],
        "Random Forest": [],
        "Gradient Boosting": [],
        "MLP": []
    }

    results = {
        "Linear Regression": [],
        "Random Forest": []
    }

    for past_windows in past_windows_possibilities:
        #generate_X_and_Y(None)

        # Load the dataframes
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = preprocess()

        # Train and evaluate the baseline models
        baseline_models = BaselineModels()

        maes = baseline_models.train_and_evaluate(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler, past_windows=past_windows)

        for model_name, mae in maes.items():
            results[model_name].append(mae)

        plot_error_vs_past_windows(results)

def experiment_all_models_results_vs_past_windows_number():

    from preprocessing import preprocess
    from models import BaselineModels
    from Data.generate_x_and_y import generate_X_and_Y
    from ploting import plot_error_vs_past_windows
    from pipeline import pipeline

    # All possibilities of number of past_windows to experiment with
    past_windows_possibilities = list(range(8, 15))

    results = {
        'ANN': [],
        'LSTM': [],
        'GRU': [],
        'TCN': [],
        'CNN': [],
        'DCNN': [],
        'CNNLSTM': [],
        "Linear Regression": [],
        "Random Forest": [],
        "Gradient Boosting": [],
        "MLP": []
    }

    for past_windows in past_windows_possibilities:
        generate_X_and_Y(past_windows)

        # Load the dataframes
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = preprocess(randomize=True)

        baseline_models = BaselineModels()

        maes = baseline_models.train_and_evaluate(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler, past_windows)

        for model_name, mae in maes.items():
            if model_name in ['Random Forest', 'MLP']:
                results[model_name].append(mae)

        for model_name in ['TCN', 'GRU', 'ANN']:
            try:
                Y, Y_pred, error_all, error_test = pipeline(model_name, num_past_windows=past_windows, X_train=X_train_scaled, X_test=X_test_scaled, Y_train=Y_train_scaled, Y_test=Y_test_scaled, Y_scaler=Y_scaler)
                results[model_name].append(error_test)
            except:
                results[model_name].append(-0.001)
            

        plot_error_vs_past_windows(results)

def main():

    experiment_baselineModels_results_vs_past_windows_number()

    #experiment_all_models_results_vs_past_windows_number()

if __name__ == "__main__":
    main()