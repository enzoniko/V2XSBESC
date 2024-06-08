
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ploting import plot_results

# Simple models to use as a baseline
class BaselineModels:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor()),
            "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
        }

    def reshape_to_2d(self, X, Y):
        
        # If X shape is 3D, reshape it to 2D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        return X, Y
    
    def train_model(self, model, X_train, Y_train):

        model.fit(X_train, Y_train)
        return model

    def evaluate_model(self, model, X_test, Y_test):

        score = model.score(X_test, Y_test)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        return score, predictions, mae, mse

    def train_and_evaluate(self, X_train, X_test, Y_train, Y_test, Y_scaler, past_windows):

        X_train, Y_train = self.reshape_to_2d(X_train, Y_train)
        X_test, Y_test = self.reshape_to_2d(X_test, Y_test)

        Y_test_descaled = Y_scaler.inverse_transform(Y_test)

        maes = {}

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model = self.train_model(model, X_train, Y_train)
            score, predictions, mae, mse = self.evaluate_model(model, X_test, Y_test)
            print(f"{model_name} - Score: {score}, MSE: {mse}, MAE: {mae}")

            # Descale the data
            predictions = Y_scaler.inverse_transform(predictions)
            

            plot_results(Y_test_descaled, predictions, model_name, mae, mse, past_windows)

            maes[model_name] = mae

        return maes