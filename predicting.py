from preprocessing import descale_data
import numpy as np
# Import mean percentage error function
from keras.losses import mean_absolute_error


def make_predictions(model, X_train, X_test, Y_train, Y_test, Y_scaler):
    
    # Get the predictions
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    if Y_train_pred.shape[1] == 1:
        # Reshape Y_train_pred and Y_test_pred to be 2D
        Y_train_pred = Y_train_pred.reshape(Y_train_pred.shape[0], Y_train_pred.shape[2])
        Y_test_pred = Y_test_pred.reshape(Y_test_pred.shape[0], Y_test_pred.shape[2])

    # print(Y_test_pred[:10])

    print("Shape Y_train_pred:", Y_train_pred.shape)
    print("Shape Y_test_pred:", Y_test_pred.shape)
        
    # Denormalize the predictions
    Y_train_pred_descaled = descale_data(Y_train_pred, Y_scaler)
    Y_test_pred_descaled = descale_data(Y_test_pred, Y_scaler)

    print("Shape Y_train:", Y_train.shape)
    print("Shape Y_test:", Y_test.shape)

    # Denormalize the Y values
    Y_train_descaled = descale_data(Y_train, Y_scaler)
    Y_test_descaled = descale_data(Y_test, Y_scaler)
    
    # Concatenate the Y values from the training and test sets
    Y = np.concatenate((Y_train, Y_test), axis=0)

    # Concatenate the predictions from the training and test sets
    Y_pred = np.concatenate((Y_train_pred, Y_test_pred), axis=0)

    # Check for NaN values in the Y and Y_pred
    print("NaN values in Y:", np.isnan(Y).any())
    print("NaN values in Y_pred:", np.isnan(Y_pred).any())

    # Check for inf values in the Y and Y_pred
    print("Inf values in Y:", np.isinf(Y).any())
    print("Inf values in Y_pred:", np.isinf(Y_pred).any())

    # Check for zeros in the Y and Y_pred
    print("Zeros in Y:", (Y == 0).any())
    print("Zeros in Y_pred:", (Y_pred == 0).any())

    # Get the mean percentage error for the train set
    error_train = mean_absolute_error(Y_train, Y_train_pred).numpy()

    error_train = np.mean(error_train)
    print("Mean percentage error for the train set:", error_train)

    # Get the mean percentage error for all
    error = mean_absolute_error(Y, Y_pred).numpy()

    error = np.mean(error)

    print('Mean absolute percentage error for the entire dataset:', error)

    # Get the mean percentage error for the test set
    error_test = mean_absolute_error(Y_test, Y_test_pred).numpy()

    print("SHAPE ERROR", error_test.shape)
    error_test_std = np.std(error_test)

    print("STD DEV ERROR TEST:", error_test_std)
    error_test = np.mean(error_test)
   

    return Y, Y_pred, Y_test_descaled, Y_test_pred_descaled, error, error_test, error_test_std