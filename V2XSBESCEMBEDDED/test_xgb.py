from xgboost import XGBRegressor
import joblib
import numpy as np
import time

if __name__ == '__main__':
    # Load the model
    model = XGBRegressor(max_depth=3, n_estimators=100)
    
    # Load the model from exported/XGB/xgb_model.pkl
    model.load_model("exported/XGB/xgb_model.txt")
    # Load the test data
    X_test = np.loadtxt("exported/XGB/X_test.txt")
    Y_test = np.loadtxt("exported/XGB/Y_test.txt")

    initial_time = time.time()
    # Predict
    Y_pred = model.predict(X_test)

    print("Time taken to predict:", time.time() - initial_time)

    # Run the prediction 10000 times and record the time taken for each prediction
    prediction_times = []
    for _ in range(10000):
        start_time = time.time()
        model.predict(X_test)
        end_time = time.time()
        prediction_times.append(end_time - start_time)

    # Calculate the average and standard deviation of prediction times
    average_time = np.mean(prediction_times)
    std_dev_time = np.std(prediction_times)

    print("Average prediction time:", average_time)
    print("Standard deviation of prediction time:", std_dev_time)

    # Calculate MAE
    mae = np.mean(np.abs(Y_test - Y_pred))

    print("MAE for XGB model:", mae)
