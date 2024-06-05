
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

    def train_model(self, model, X_train, Y_train):
        model.fit(X_train, Y_train)
        return model

    def evaluate_model(self, model, X_test, Y_test):
        score = model.score(X_test, Y_test)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        return score, predictions, mae, mse

    def train_and_evaluate(self, X_train, X_test, Y_train, Y_test):
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model = self.train_model(model, X_train, Y_train)
            score, predictions, mae, mse = self.evaluate_model(model, X_test, Y_test)
            print(f"{model_name} - Score: {score}, MSE: {mse}, MAE: {mae}")
            plot_results(Y_test, predictions, model_name, mae, mse)