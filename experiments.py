
import pickle

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



""" parameters = {

    'TCN': {
        'num_filters': [16, 32, 64],
        'kernel_size': [2, 3, 4],
        'l2': [0.1, 0.01, 0.001],
        'dense_units': [10, 50, 100]
    },
    'GRU': {
        'units': [10, 50, 100, 150, 200],
        'l2': [0.1, 0.01, 0.001],

    },
    'ANN': {
        'units': [10, 30, 50, 70, 100],
        'l2': [0.1, 0.01, 0.001]
    },
    'CNN': {
        'filters': [16, 32, 64],
        'l2': [0.1, 0.01, 0.001]
    }
} """

parameters = {

    'TCN': {
        'num_filters': [16],
        'kernel_size': [2],
        'l2': [0.1],
        'dense_units': [10, 50]
    },
    'GRU': {
        'units': [10],
        'l2': [0.1, 0.01],

    },
    'ANN': {
        'units': [10],
        'l2': [0.001]
    }
}

from itertools import product

# Function to generate all combinations of hyperparameters
def generate_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combination)) for combination in product(*values)]


def experiment_all_models_results_vs_past_windows_number():

    from preprocessing import preprocess
    from models import BaselineModels
    from Data.generate_x_and_y import generate_X_and_Y
    from ploting import plot_error_vs_past_windows
    from pipeline import pipeline

    # All possibilities of number of past_windows to experiment with
    past_windows_possibilities = list(range(1, 6))

    """ results = {
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
    } """

    results = {
        'Random Forest': [],
        'XGBoost': [],
        'Moving Average': [],
        #'GRU': [],
        'ANN': [],
        #'CNN': []
    }

    best_parameters = {
        'Random Forest': [],
        'XGBoost': [], 
        'Moving Average': [],
        #'GRU': [],
        'ANN': [],
        #'CNN': []
    }

    for past_windows in past_windows_possibilities:
        mae_moving_average = generate_X_and_Y(past_windows)

        # Load the dataframes
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = preprocess(randomize=True)

        baseline_models = BaselineModels()

        # Adapted BaselineModels to only run Grid Search with RF, returning the best_parameters too
        maes, best_parameters_all = baseline_models.train_and_evaluate(X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler, past_windows)

        for model_name, mae in maes.items():
            if model_name in ['Random Forest', 'XGBoost']:
                results[model_name].append(mae)

        for model_name, best_params in best_parameters_all.items():
            if model_name in ['Random Forest', 'XGBoost']:
                best_parameters[model_name].append((best_params, past_windows, maes[model_name]))
       
        for model_name in ['ANN']:
            param_grid = parameters[model_name]
            combinations = generate_combinations(param_grid)
            best_error = float('inf')
            best_params = None

            for params in combinations:
                try:
                    Y, Y_pred, error_all, error_test, error_test_std = pipeline(model_name, 
                                                                num_past_windows=None, 
                                                                X_train=X_train_scaled, 
                                                                X_test=X_test_scaled, 
                                                                Y_train=Y_train_scaled, 
                                                                Y_test=Y_test_scaled, 
                                                                Y_scaler=Y_scaler, 
                                                                model_params = params,
                                                                use_kfold=False)
                
                    if error_test < best_error:
                        best_error = error_test
                        best_params = params
                except Exception as e:
                    print(e)
                    continue

            results[model_name].append(best_error)
            best_parameters[model_name].append((best_params, past_windows, best_error))
            
        # Save the moving average results too
        results['Moving Average'].append(mae_moving_average)
        best_parameters['Moving Average'].append((None, past_windows, mae_moving_average))

        plot_error_vs_past_windows(results)

    # Save the best_parameters structure to disk
    with open('best_parameters.pkl', 'wb') as f:
        pickle.dump(best_parameters, f)

    with open('best_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Best parameters have been saved to best_parameters.pkl")

def main():

    #experiment_baselineModels_results_vs_past_windows_number()

    experiment_all_models_results_vs_past_windows_number()

if __name__ == "__main__":
    main()