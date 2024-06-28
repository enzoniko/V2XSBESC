import emlearn
import numpy as np
import pandas as pd
import joblib
import os.path

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from callbacks import assemble_callbacks

from keras.models import Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# EMLEARN STUFF

def convert_model(model, name, X_test, Y_test):


    model_filename = os.path.join(os.path.join(os.path.dirname(__file__), 'exporting/RF'), f'{name}.h')
    print(model_filename)
    cmodel = emlearn.convert(model, return_type='regressor', method='loadable')
    code = cmodel.save(file=model_filename, name=name)

    test_pred = cmodel.predict(X_test)

    print(Y_test[:5])
    print(test_pred[:5])

    mae = mean_absolute_error(Y_test, test_pred)
    print(f'MAE for converted model {name} is {mae}')

    assert os.path.exists(model_filename)
    print(f"Generated {model_filename}")

def build_ANN_model_flat(input_shape, units=10, **kwargs):
        model = Sequential()
        model.add(Dense(units, activation='relu', input_shape=input_shape))
        model.add(Dense(2))
        adam = Adam(learning_rate=0.0001)
        model.compile(optimizer=adam, loss='mae')
        return model


def export_model(model_name):

    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')

    print(f"Shape of X_test is {X_test.shape} and Y_test is {Y_test.shape}")

    print('First element of X and Y test')
    print(X_test[0])
    print(Y_test[0])

    

    if model_name == 'ANN':
        callbacks = assemble_callbacks('ANN')

        # Flatten X 
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Save flattened test data to text files
        np.savetxt("exporting/ANN/X_test.txt", X_test)
        np.savetxt("exporting/ANN/Y_test.txt", Y_test)

        model = build_ANN_model_flat(input_shape=(X_train.shape[1],))

        # To train the model
        model.fit(X_train, Y_train, epochs=1500, callbacks=callbacks)
        print(f"Model evaluation result: {model.evaluate(X_test, Y_test)}")

        # To use a trained model instead
        #model.load_weights('Models/ANNbest_model.h5')

        # Export the weights
        for layer in model.layers:
            weights = layer.get_weights()
            for i, w in enumerate(weights):
                np.savetxt(f"exporting/ANN/{layer.name}_weight_{i}.txt", w.T) # SAVE THE WEIGHTS TRANSPOSED!

        # CODE TO INSPECT HIDDEN LAYER FOR DEBUG REASONS
        """ layer_outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)

        # Print activations of a specific hidden layer for the first 5 entries
        num_samples_to_print = 5
        hidden_layer_index = 0  # Example: index 1 corresponds to the first hidden layer

        # Iterate through the first 5 test entries
        for i in range(min(num_samples_to_print, X_test.shape[0])):
            input_sample = X_test[i:i+1]  # Select the i-th sample (reshape if needed)
            activations = activation_model.predict(input_sample)

            # Extract activations of the specified hidden layer
            hidden_layer_activations = activations[hidden_layer_index]

            print(f"Sample {i + 1} - Keras Hidden Layer {hidden_layer_index + 1} Activations:")
            print(hidden_layer_activations)
            print()  # Print a blank line for separation

        print("-----------------------------------") """

        # Predict and show the first 5 actual vs predicted values
        predicted = model.predict(X_test)
        print(predicted[:5])
        print(Y_test[:5])
   
    elif model_name == 'RF':
        
        # Save flattened test data to text files
        np.savetxt("exporting/RF/X_test.txt", X_test.reshape(X_test.shape[0], -1))
        np.savetxt("exporting/RF/Y_test.txt", Y_test.reshape(X_test.shape[0], -1))
        
        Y_train_feature1 = Y_train[:, 0]  # First feature of Y
        Y_train_feature2 = Y_train[:, 1]  # Second feature of Y
        
        # Create RandomForestRegressor for feature 1
        rf_feature1 = RandomForestRegressor(n_jobs=24, n_estimators=50, max_depth=10, oob_score=True)
        rf_feature1.fit(X_train.reshape(X_train.shape[0], -1), Y_train_feature1) 

        # Create RandomForestRegressor for feature 2
        rf_feature2 = RandomForestRegressor(n_jobs=24, n_estimators=50, max_depth=10, oob_score=True)
        rf_feature2.fit(X_train.reshape(X_train.shape[0], -1), Y_train_feature2)

        # Predictions for feature 1
        Y_pred_feature1 = rf_feature1.predict(X_test.reshape(X_test.shape[0], -1))

        # Predictions for feature 2
        Y_pred_feature2 = rf_feature2.predict(X_test.reshape(X_test.shape[0], -1))

        # Calculate MAE for feature 1 predictions
        mae_feature1 = mean_absolute_error(Y_test[:, 0], Y_pred_feature1)

        # Calculate MAE for feature 2 predictions
        mae_feature2 = mean_absolute_error(Y_test[:, 1], Y_pred_feature2)

        print("MAE for feature 1:", mae_feature1)
        print("MAE for feature 2:", mae_feature2)

        print("Converting RF models, this takes a long time")
        convert_model(rf_feature1, 'rf1', X_test.reshape(X_test.shape[0], -1), Y_test[:, 0])
        convert_model(rf_feature2, 'rf2', X_test.reshape(X_test.shape[0], -1), Y_test[:, 1])
        
if __name__ == '__main__':

    # Export ANN embedded model stuff
    #export_model('ANN')

    # Export RF embedded model stuff
    export_model('RF')
    
    