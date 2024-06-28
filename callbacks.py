import numpy as np
from keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping, ModelCheckpoint

class TerminateNaN(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is not None and val_loss is not None:
            if np.isnan(loss) or np.isinf(loss) or np.isnan(val_loss) or np.isinf(val_loss):
                print("Invalid loss and terminated at epoch {}".format(epoch))
                self.model.stop_training = True

def assemble_callbacks(name):
    # Save the best model during training
    model_checkpoint = ModelCheckpoint(f'Models/{name}best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Stop training when the loss or validation loss is not valid
    terminate_nan = TerminateNaN()

    # Stop training when the validation loss has stopped decreasing for 15 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)

    # Reduce learning rate when the validation loss has stopped decreasing for 5 epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

    return [reduce_lr, terminate_nan, early_stopping, model_checkpoint]