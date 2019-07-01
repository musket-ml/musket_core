from keras.callbacks import *
import configloader


class CallbackModule(Callback):

    def on_train_begin_action(self, logs = None):
        pass

    def on_epoch_begin_action(self, epoch, logs = None):
        pass

    def on_epoch_end_action(self, epoch, logs = None):
        pass

    def on_batch_begin_action(self, batch, logs = None):
        pass

    def on_batch_end_action(self, batch, logs = None):
        pass

    def on_train_begin(self, logs=None):
        if self.then is not None:
            self.then.on_train_begin(logs)
        if self.startStep == 0:
            self.on_train_begin_action()

    def on_epoch_end(self, epoch, logs=None):
        if self.is_applicable():
            self.on_epoch_end_action(epoch,logs)
        if self.then is not None:
            self.then.on_epoch_end(epoch,logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.batch = 0
        self.recalc_position()
        if self.is_applicable():
            self.on_epoch_begin_action(epoch,logs)
        if self.then is not None:
            self.then.on_epoch_begin(epoch,logs)

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        self.recalc_position()
        if self.is_applicable():
            self.on_batch_begin_action(batch,logs)
        if self.then is not None:
            self.then.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        if self.is_applicable():
            self.on_batch_end_action(batch,logs)
        if self.then is not None:
            self.then.on_batch_end(batch,logs)


    def __init__(self, **args):
        super(CallbackModule, self).__init__()

        self.relSize = args['relSize'] if 'relSize' in args else None
        self.absSize = args['absSize'] if 'absSize' in args else None
        if self.absSize is None and self.relSize is None:
            raise ValueError("'absSize' or 'relSize' must be specified for CallbackModule")
        if self.absSize is not None and self.relSize is not None:
            raise ValueError("'absSize' and 'relSize' are mutually exclusive for CallbackModule")

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

    def get_then_params(self):
        then_params = self.params.copy()
        then_params['startStep'] = self.endStep
        return then_params

    def set_model(self, model):
        self.model = model
        if self.then is not None:
            self.then.set_model(model)

    def is_applicable(self):
        return self.step >= self.startStep and self.step < self.endStep

    def recalc_position(self):
        self.step = self.epochSize * self.epoch + self.batch
