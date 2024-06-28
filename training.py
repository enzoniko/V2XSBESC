import sys
from sklearn.model_selection import KFold
from keras.callbacks import History
from models import *
import os

def train_with_kfold(model_name, X_train, Y_train, X_test, Y_test, callbacks, use_kfold=True, model_parameters=None) -> tuple:

    # Redirect the output to a log file
    sys.stdout = open(f'Logs/{model_name}_training.log', 'a')

    # Create a KFold object with 5 splits
    kf = KFold(n_splits=5, shuffle=False)

    # Initialize a list to store the results of each fold (to calculate the average result)
    results = []

    # Initialize a history object to store the history of all folds (to plot the loss of all folds together)
    all_history = History()
    all_history.history = {'loss': [], 'val_loss': []}

    # If there is a file called 'Models/<model_name>_best_model_final.h5', delete it (to save the best model)
    if os.path.isfile(f'Models/{model_name}/best_model_final.h5'):
        os.remove(f'Models/{model_name}/best_model_final.h5')

    # If the use_kfold parameter is False, just use one fold
    split = [[[train_index, val_index] for train_index, val_index in kf.split(X_train)][0]] if not use_kfold else kf.split(X_train)

    # Loop over each fold
    for train_index, val_index in split:
        # Split the data into training and validation sets of the current fold
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

        # Build the model
        model = build_model(model_name, X_train.shape[1:], **model_parameters)

        print(model.summary())

        # Train the model on the current fold and store the history
        history = model.fit(X_train_fold, Y_train_fold, validation_data=(X_val_fold, Y_val_fold), epochs=1500, verbose=1, callbacks=callbacks)

        # Concatenate the history
        all_history.history['loss'].extend(history.history['loss'])
        all_history.history['val_loss'].extend(history.history['val_loss'])

        # Evaluate the model on the test set
        model.load_weights(f'Models/{model_name}/best_model.h5')
        result = model.evaluate(X_test, Y_test)
        print("Evaluation result on current fold:", result)

        # Verify if there is a file called 'Models/<model_name>/best_model_final.h5'
        if os.path.isfile(f'Models/{model_name}/best_model_final.h5'):
            # If there is, load it in a separate model to compare the results
            model_final = build_model(model_name, X_train.shape[1:])
            model_final.load_weights(f'Models/{model_name}/best_model_final.h5')
            result_final = model_final.evaluate(X_test, Y_test)
            # If the current fold model result is better than the best model, save the current model as the best model
            if result < result_final:
                model.save_weights(f'Models/{model_name}/best_model_final.h5')
        # If there isn't, save the model as the best model
        else:
            model.save_weights(f'Models/{model_name}/best_model_final.h5')

        # Append the result to the results list
        results.append(result)

    return results, all_history