from keras.callbacks import *
from musket_core import configloader

class CallbackModule(Callback):
    """
        # Example
            ```yaml
                  LRVariator:
                    absSize: 100
                    toVal: 0.002
                    style: 2
                    then:
                      LRVariator:
                        relSize: 0.001
                        toVal: 0.001
                        style: 0.5
                        then:
                          ReduceLROnPlateau:
                            patience: 8
                            factor: 0.5
                            monitor: val_binary_accuracy
                            mode: auto
                            cooldown: 5
                            verbose: 1
            ```

        # Arguments
            relSize: activity time expressed in epochs, mutually exclusive with 'absSize'
            absSize: activity time expressed in batches, mutually exclusive with 'relSize'
            periodEpochs: cycle length expressed in epochs, omited value means no cycle, mutually exclusive with 'periodSteps'
            periodSteps: cycle length expressed in epochs, omited value means no cycle,  mutually exclusive with 'periodEpochs'
            then: callback to be applied after activity time elapses. The callback inherits 'periodEpochs' and 'periodSteps' values
        """

    def on_train_begin_action(self, logs = None):
        pass

    def on_epoch_begin_action(self, logs = None):
        pass

    def on_epoch_end_action(self, logs = None):
        pass

    def on_batch_begin_action(self, logs = None):
        pass

    def on_batch_end_action(self, logs = None):
        pass

    def on_train_begin(self, logs=None):
        if self.then is not None:
            self.then.on_train_begin(logs)
        if self.startStep == 0:
            self.on_train_begin_action(logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.is_applicable():
            self.on_epoch_end_action(logs)
        if self.then is not None:
            self.then.on_epoch_end(epoch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.batch = 0
        self.recalc_position()
        if self.is_applicable():
            self.on_epoch_begin_action(logs)
        if self.then is not None:
            self.then.on_epoch_begin(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        self.recalc_position()
        if self.is_applicable():
            self.on_batch_begin_action(logs)
        if self.then is not None:
            self.then.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        if self.is_applicable():
            self.on_batch_end_action(logs)
        if self.then is not None:
            self.then.on_batch_end(batch, logs)


    def __init__(self, **args):
        super(CallbackModule, self).__init__()

        self.relSize = args['relSize'] if 'relSize' in args else None
        self.absSize = args['absSize'] if 'absSize' in args else None
        self.periodEpochs = args['periodEpochs'] if 'periodEpochs' in args else None
        self.periodSteps = args['periodSteps'] if 'periodSteps' in args else None

        if self.absSize is None and self.relSize is None:
            raise ValueError("'absSize' or 'relSize' must be specified for CallbackModule")
        if self.absSize is not None and self.relSize is not None:
            raise ValueError("'absSize' and 'relSize' are mutually exclusive for CallbackModule")
        if self.periodEpochs is not None and self.periodSteps is not None:
            raise ValueError("'periodEpochs' and 'periodSteps' are mutually exclusive for CallbackModule")

        self.then = None
        if 'then' in args:
            thenArg = args['then']
            parsedThen = configloader.parse("callbacks", thenArg)
            if len(parsedThen) > 0:
                self.then = parsedThen[0]


    def set_params(self, params):
        self.params = params
        self.init()
        if self.then is not None:
            then_params = self.get_then_params()
            self.then.set_params(then_params)

    def init(self):
        self.epochSize = self.params['steps']
        if 'startStep' in self.params:
            self.startStep = self.params['startStep']
        else:
            self.startStep = 0

        if self.absSize is not None:
            self.ownSteps = self.absSize
        elif self.relSize is not None:
            self.ownSteps = int(self.relSize * self.epochSize + 0.5)
        else:
            raise ValueError("'absSize' or 'relSize' must be provided by CallbackModule")

        self.endStep = self.startStep + self.ownSteps

        self.period = None
        if 'period' in self.params:
            self.period = self.params['period']
        elif self.periodEpochs is not None:
            self.period = self.epochSize * self.periodEpochs
        elif self.periodSteps is not None:
            self.period = self.periodSteps

    def get_then_params(self):
        then_params = self.params.copy()
        then_params['startStep'] = self.endStep
        if self.period is not None:
            then_params['period'] = self.period

        return then_params

    def set_model(self, model):
        self.model = model
        if self.then is not None:
            self.then.set_model(model)

    def is_applicable(self):
        return self.cyclicStep >= self.startStep and self.cyclicStep < self.endStep

    def recalc_position(self):
        self.step = self.epochSize * self.epoch + self.batch
        self.cyclicStep = self.step if self.period is None else self.step % self.period

class ModelCheckpointTF2(ModelCheckpoint):
    def __init__(self, *args):
        super(ModelCheckpoint, self).__init__(*args)
    
    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True, save_format='h5')
                        else:
                            self.model.save(filepath, overwrite=True, save_format='h5')
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            self._maybe_remove_file()
