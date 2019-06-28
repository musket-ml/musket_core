from keras.callbacks import *
import configloader
"""
MIT License

Copyright (c) 2017 Bradley Kenstler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class WarmupLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base=0.001, max=0.006, relSize:float=50, then: dict=None):
        super(WarmupLR, self).__init__()

        self.base_lr = base
        self.max_lr = max
        self.relSize = relSize
        self.then = None
        if then is not None:
            parsedThen = configloader.parse("callbacks", then)
            if len(parsedThen) > 0:
                self.then = parsedThen[0]

        self.step = 0
        self.lastEpoch = int(self.relSize)



    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if self.then is not None:
            self.then.on_epoch_begin(epoch,logs)


    def on_batch_begin(self, batch, logs=None):
        if self.then is not None:
            self.then.on_batch_begin(batch, logs)

    def set_params(self, params):
        self.params = params
        self.epochSize = self.params['steps']
        self.off = self.epochSize * (1 + self.lastEpoch) - self.own_steps()
        if self.then is not None:
            self.then.set_params(params)

    def set_model(self, model):
        self.model = model
        if self.then is not None:
            self.then.set_model(model)

    def get_lr(self):
        steps = self.relSize * self.epochSize
        lr = self.base_lr + (self.max_lr - self.base_lr) * self.step / steps
        return lr

    def is_applicable(self):
        steps = self.own_steps()
        return self.step < steps

    def own_steps(self):
        return int(self.relSize * self.epochSize + 0.5)

    def on_train_begin(self, logs=None):
        self.step = 0
        K.set_value(self.model.optimizer.lr, self.get_lr())

    def on_epoch_end(self, epoch, logs=None):
        if self.is_applicable():
            pass
        else:
            self.then.on_epoch_end(epoch,logs)

    def on_batch_end(self, epoch, logs=None):

        if self.is_applicable():
            K.set_value(self.model.optimizer.lr, self.get_lr())
            print(f"lr: {self.get_lr()}")
            self.step += 1
        elif self.then is not None:
            self.then.on_batch_end(epoch, logs)
        else:
            steps1 = self.relSize * self.epochSize
            steps = abs(1-self.relSize) * self.epochSize
            lr = abs(self.max_lr - (self.max_lr - self.base_lr) * (self.step-steps1) / steps)
            K.set_value(self.model.optimizer.lr, lr)
            print(f"lr: {lr}")
            self.step += 1
