
""" 
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

 """

import pickle
from ploting import plot_error_vs_past_windows
import pickle
from ploting import plot_error_vs_past_windows
def main():


    # Load the best_results from disk
    with open('best_results.pkl', 'rb') as f:
        best_results = pickle.load(f)

    # Load the best_parameters from disk
    with open('best_parameters.pkl', 'rb') as f:
        best_parameters = pickle.load(f)

    # Load the best_results_all from disk
    with open('best_results_merged.pkl', 'rb') as f:
        best_results_all = pickle.load(f)

    # Load the best_parameters_all from disk
    with open('best_parameters_merged.pkl', 'rb') as f:
        best_parameters_all = pickle.load(f)

    # Merge the dictionaries for best_results
    merged_best_results = {
        'Random Forest': best_results_all.get('Random Forest', {}),
        'GRU': best_results_all.get('GRU', {}),
        'MLP': best_results_all.get('ANN', {}),
        'TCN': best_results_all.get('TCN', {}),
        'CNN': best_results.get('CNN', {}),
        'XGBoost': best_results_all.get('XGBoost', {})
    }

    # Merge the dictionaries for best_parameters
    merged_best_parameters = {
        'Random Forest': best_parameters_all.get('Random Forest', []),
        'GRU': best_parameters_all.get('GRU', []),
        'MLP': best_parameters_all.get('ANN', []),
        'TCN': best_parameters_all.get('TCN', []),
        'CNN': best_parameters.get('CNN', []),
        'XGBoost': best_parameters_all.get('XGBoost', [])
    }

    # Print the merged results for each model
    print("MERGED BEST PARAMETERS: \n")
    for model, parameters in merged_best_parameters.items():
        print(f"{model}:")
        for paremeter in parameters:
            print(f"Past windows: {paremeter[1]}, Parameters: {paremeter[0]}, MAE: {paremeter[2]}")

    print('MERGED BEST RESULTS: \n')
    for model, results in merged_best_results.items():
        print(f"{model}: {results}")

    plot_error_vs_past_windows(merged_best_results)

    # Save the merged structures to new .pkl files
    with open('best_parameters_all.pkl', 'wb') as f:
        pickle.dump(merged_best_parameters, f)

    with open('best_results_all.pkl', 'wb') as f:
        pickle.dump(merged_best_results, f)

  
if __name__ == "__main__":
    main()