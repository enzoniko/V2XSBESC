from pipeline import pipeline
from preprocessing import preprocess
from Data.generate_x_and_y import generate_X_and_Y
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from ploting import plot_error_histograms

def reshape_to_2d(X, Y):
        # If X shape is 3D, reshape it to 2D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return X, Y


def main():

    generate_X_and_Y(1)

    # Load the dataframes
    X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_scaler = preprocess(randomize=True)

    # Run the pipeline for the ANN model
    Y_test, Y_test_pred_ANN, mae_ANN_train, mae_ANN_test, mae_ANN_test_std = pipeline('ANN', X_train=X_train_scaled, X_test=X_test_scaled, Y_train=Y_train_scaled, Y_test=Y_test_scaled, Y_scaler=Y_scaler, return_test_preds=True, num_past_windows=None)

    # Reshape the data to 2D
    X_train_scaled, Y_train_scaled = reshape_to_2d(X_train_scaled, Y_train_scaled)
    X_test_scaled, Y_test_scaled = reshape_to_2d(X_test_scaled, Y_test_scaled)

    # Run MultiOutputRegressor with RandomForestRegressor with 5 depth and 50 estimators
    mor = MultiOutputRegressor(RandomForestRegressor(max_depth=5, n_estimators=50))
    mor.fit(X_train_scaled, Y_train_scaled)
    Y_train_pred_RF = mor.predict(X_train_scaled)
    Y_test_pred_RF = mor.predict(X_test_scaled)

    # Calculate the Mean Absolute Error for the Random Forest model for train and test
    mae_RF_train = mean_absolute_error(Y_train_scaled, Y_train_pred_RF)
    mae_RF_test = mean_absolute_error(Y_test_scaled, Y_test_pred_RF)

    # Calculate the standard deviation of the Mean Absolute Error for the Random Forest model for test
    mae_RF_test_std = abs(Y_test_scaled - Y_test_pred_RF).std()

    # Run MultiOutputRegressor with XGBoosTRegressor with 5 depth and 100 estimators and a learning rate of 0.1
    mor = MultiOutputRegressor(XGBRegressor(max_depth=5, n_estimators=100, learning_rate=0.1))
    mor.fit(X_train_scaled, Y_train_scaled)
    Y_train_pred_XGB = mor.predict(X_train_scaled)
    Y_test_pred_XGB = mor.predict(X_test_scaled)

    # Calculate the Mean Absolute Error for the XGBoost model for train and test
    mae_XGB_train = mean_absolute_error(Y_train_scaled, Y_train_pred_XGB)
    mae_XGB_test = mean_absolute_error(Y_test_scaled, Y_test_pred_XGB)

    # Calculate the standard deviation of the Mean Absolute Error for the XGBoost model for test
    mae_XGB_test_std = abs(Y_test_scaled - Y_test_pred_XGB).std()

    # Print the results
    print(f"ANN MAE TRAIN: {mae_ANN_train}")
    print(f"ANN MAE TEST: {mae_ANN_test}")
    print(f"ANN MAE TEST STD DEV: {mae_ANN_test_std}")

    print(f"RF MAE TRAIN: {mae_RF_train}")
    print(f"RF MAE TEST: {mae_RF_test}")
    print(f"RF MAE TEST STD DEV: {mae_RF_test_std}")

    print(f"XGB MAE TRAIN: {mae_XGB_train}")
    print(f"XGB MAE TEST: {mae_XGB_test}")
    print(f"XGB MAE TEST STD DEV: {mae_XGB_test_std}")

    # Inverse the scaling of the predictions
    Y_test_pred_RF = Y_scaler.inverse_transform(Y_test_pred_RF)
    Y_test_pred_XGB = Y_scaler.inverse_transform(Y_test_pred_XGB)

    # Plot the results
    plot_error_histograms({
        'ANN': (Y_test, Y_test_pred_ANN, mae_ANN_train, mae_ANN_test, mae_ANN_test_std),
        'RF': (Y_test, Y_test_pred_RF, mae_RF_train, mae_RF_test, mae_RF_test_std),
        'XGB': (Y_test, Y_test_pred_XGB, mae_XGB_train, mae_XGB_test, mae_XGB_test_std)
    })







    
if __name__ == '__main__':
    main()