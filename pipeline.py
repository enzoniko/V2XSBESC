from preprocessing import preprocess
from callbacks import assemble_callbacks
from Data.generate_x_and_y import generate_X_and_Y
from models import build_model
from training import train_with_kfold
from predicting import make_predictions
from ploting import plot_results

import os
import sys
from time import time
import numpy as np

def pipeline(model_name, use_kfold=False, return_test_preds=True, num_past_windows=None, X_train=None, X_test=None, Y_train=None, Y_test=None, Y_scaler=None, model_params={}):

    if num_past_windows:
        # Generate the data
        generate_X_and_Y(num_past_windows)

    # Get the preprocessed data
    if X_train is None:
        X_train, X_test, Y_train, Y_test, Y_scaler = preprocess()

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('Y_train.npy', Y_train)
    np.save('Y_test.npy', Y_test)
    

    # Print the shape of the data
    print(f"X_train_scaled shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    # If the Logs folder does not exist, create it
    if not os.path.exists('Logs'):
        os.makedirs('Logs')

    # Clear the log file
    open(f'Logs/{model_name}_training.log', 'w').close()

    # If the Models folder does not exist, create it
    if not os.path.exists('Models'):
        os.makedirs('Models')

    # If the model_name folder does not exist, create it
    if not os.path.exists(f'Models/{model_name}'):
        os.makedirs(f'Models/{model_name}')

    # If there is a file called 'Models/<model_name>/best_model_final.h5', delete it (to save the best model)
    if os.path.isfile(f'Models/{model_name}/best_model_final.h5'):
        os.remove(f'Models/{model_name}/best_model_final.h5')
    
    # Create the callbacks
    callbacks = assemble_callbacks(f'{model_name}/')

    # Mark the start time of the training
    start_time = time()

    stdout = sys.stdout

    # Train the model with KFold
    results, all_history = train_with_kfold(model_name, X_train, Y_train, X_test, Y_test, callbacks, use_kfold, model_parameters=model_params)

    # Mark the end time of the training
    end_time = time()

    # Calculate the average result
    average_result = sum(results) / len(results)
    print("Average evaluation result of the folds:", average_result)
    print("Results of the folds:", results)

    # Load the best model
    model = build_model(model_name, X_train.shape[1:], **model_params)
    model.load_weights(f'Models/{model_name}/best_model_final.h5')

    # Make predictions
    Y, Y_pred, Y_test_descaled, Y_test_pred_descaled, error, error_test, error_test_std = make_predictions(model, X_train, X_test, Y_train, Y_test, Y_scaler)

    print("Shape of Y_test_descaled:", Y_test_descaled.shape)
    print("Shape of Y_test_pred_descaled:", Y_test_pred_descaled.shape)

    # Plot the results
    plot_results(Y_test_descaled, Y_test_pred_descaled, model_name, error, error_test, error_test_std, num_past_windows)

    # Print the time taken to train the model
    print(f"Time taken to train the model: {end_time - start_time} seconds")

    # Close the log file
    sys.stdout.close()

    sys.stdout = stdout

    if return_test_preds:
        return Y_test_descaled, Y_test_pred_descaled, error, error_test, error_test_std
    
    return Y, Y_pred, error, error_test, error_test_std


if __name__ == "__main__":

    pipeline("ANN", use_kfold=False, return_test_preds=False, num_past_windows=1)