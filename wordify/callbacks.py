
import json
from pathlib import Path

import numpy as np
import keras

def _np_to_float(x):
    if isinstance(x, np.float32):
        return float(x)
    return x


def _numpy_vals_to_floats(d):
    return {k: _np_to_float(v) for k, v in d.items()}


class NDJSONLoggingCallback(keras.callbacks.Callback):
    '''
    Log lots of ndjson.
    '''
    def __init__(self, path, log_frequency=5):
        self.log_file = path.open('w')
        self.log_frequency = log_frequency

    def record(self, logs):
        logs = _numpy_vals_to_floats(logs)
        print(json.dumps(logs), file=self.log_file, flush=True)

    def on_epoch_begin(self, epoch_n, logs=None):
        self.epoch_n = epoch_n

    def on_epoch_end(self, epoch_n, logs=None):
        logs['epoch_n'] = epoch_n
        self.record(logs)

    def on_batch_end(self, batch_n, logs=None):
        if batch_n % self.log_frequency == 0:
            logs['epoch_n'] = self.epoch_n
            logs['batch_n'] = batch_n
            self.record(logs)


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    '''
    Train one model and checkpoint another.
    Useful if you have an embedding you care about and a task you don't.
    '''
    def __init__(self, path, model, *args, **kwargs):
        self.passed_model = model
        path = Path(path).expanduser()
        path.mkdir(exist_ok=True, parents=True)
        
        if kwargs.get('save_best_only', False):
            # only save bese result...
            path = str(path / 'weights.hdf5')
        else:
            path = str(path / 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
            
        super().__init__(path, *args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.set_model(self.passed_model)
        super().on_epoch_end(epoch, logs)