
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ploting import plot_results


from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization, RepeatVector, TimeDistributed, GRU
from keras import regularizers, optimizers
from tcn import TCN
from keras.optimizers import Adam

import numpy as np


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

    def evaluate_model(self, model, X_test, Y_test, X_train, Y_train):

        score = model.score(X_test, Y_test)
        predictions = model.predict(X_test)
        mae_test = mean_absolute_error(Y_test, predictions)

        X_all = np.concatenate((X_train, X_test), axis=0)
        Y_all = np.concatenate((Y_train, Y_test), axis=0)
        predictions_all = model.predict(X_all)
        mae_all = mean_absolute_error(Y_all, predictions_all)

        return score, predictions, mae_all, mae_test

    def train_and_evaluate(self, X_train, X_test, Y_train, Y_test, Y_scaler, past_windows):

        X_train, Y_train = self.reshape_to_2d(X_train, Y_train)
        X_test, Y_test = self.reshape_to_2d(X_test, Y_test)

        Y_test_descaled = Y_scaler.inverse_transform(Y_test)

        maes = {}

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model = self.train_model(model, X_train, Y_train)
            score, predictions, mae_all, mae_test = self.evaluate_model(model, X_test, Y_test, X_train, Y_train)
            print(f"{model_name} - Score: {score}, MAE_ALL: {mae_all}, MAE_TEST: {mae_test}")

            # Descale the data
            predictions = Y_scaler.inverse_transform(predictions)
            

            plot_results(Y_test_descaled, predictions, model_name, mae_all, mae_test, past_windows)

            maes[model_name] = mae_test

        return maes
    
def build_LSTM_model(input_shape, units=50, **kwargs):
    model = Sequential()
    model.add(LSTM(units, activation='tanh', input_shape=input_shape, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(LSTM(units, activation='tanh', return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(3))
    adam = optimizers.Adam(clipvalue=0.5, learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mae')
    return model

def build_GRU_model(input_shape, units=100, l2=0.0001, **kwargs):
    model = Sequential()
    model.add(GRU(units, activation='tanh', return_sequences=True, input_shape=input_shape, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(GRU(units, activation='tanh', return_sequences=False, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(3))
    adam = optimizers.Adam(clipvalue=0.5)
    model.compile(optimizer=adam, loss='mae')
    return model

def build_TCN_model(input_shape, **kwargs):
    model = Sequential()
    model.add(TCN(nb_filters=64, kernel_size=3, activation='tanh', input_shape=input_shape, return_sequences=False))
    model.add(Dense(100, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Flatten())
    model.add(Dense(3, activation='tanh'))
    model.compile(optimizer='adam', loss='mae')
    return model

def build_DILATEDCNN_model(input_shape, filters=16, kernel_size=2, dilation=3, units=64, l2=0.12):
    model = Sequential()
    for i in range(5):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation**i, padding='causal', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(l2=l2)))
        model.add(BatchNormalization())
    model.add(Conv1D(filters=filters, kernel_size=1, activation='tanh', kernel_regularizer=regularizers.l2(l2=l2)))
    model.add(Flatten())
    for _ in range(3):
        model.add(Dense(units, activation='tanh', kernel_regularizer=regularizers.l2(l2=l2)))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mae')
    return model

def build_CNN_model(input_shape, num_layers=1, filters=30, kernel_size=2, pool_size=2, l2=0.02):
    model = Sequential()
    model.add(Conv1D(filters, kernel_size, activation='tanh', input_shape=input_shape, kernel_regularizer=regularizers.l2(l2=l2)))
    model.add(MaxPooling1D(pool_size))
    for _ in range(num_layers - 1):
        model.add(Conv1D(filters, kernel_size, activation='tanh', kernel_regularizer=regularizers.l2(l2=l2)))
        model.add(MaxPooling1D(pool_size))
    model.add(Flatten())
    model.add(Dense(3))
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mae')
    return model

def build_CNNLSTM_model(input_shape, CNN_layers=1, CNN_filters=100, CNN_kernel_size=2, CNN_pool_size=2, LSTM_layers=2, LSTM_units=200):
    model = Sequential()
    model.add(Conv1D(CNN_filters, CNN_kernel_size, activation='tanh', input_shape=input_shape, kernel_regularizer=regularizers.l2(l2=0.005)))
    model.add(MaxPooling1D(CNN_pool_size))
    for _ in range(CNN_layers - 1):
        model.add(Conv1D(CNN_filters, CNN_kernel_size, activation='tanh', kernel_regularizer=regularizers.l2(l2=0.01)))
        model.add(MaxPooling1D(CNN_pool_size))
    for _ in range(LSTM_layers - 1):
        model.add(LSTM(units=LSTM_units, input_shape=input_shape, activation="tanh", return_sequences=True))
    model.add(LSTM(units=LSTM_units, input_shape=input_shape, activation="tanh", return_sequences=False))
    model.add(Flatten())
    model.add(Dense(3))
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mae')
    return model

models_mapping = {
    'LSTM': build_LSTM_model,
    'GRU': build_GRU_model,
    'TCN': build_TCN_model,
    'CNN': build_CNN_model,
    'DCNN': build_DILATEDCNN_model,
    'CNNLSTM': build_CNNLSTM_model
}

def build_model(model_name, input_shape, **kwargs):
    return models_mapping[model_name](input_shape, **kwargs)
