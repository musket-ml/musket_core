from musket_core.callback_module import CallbackModule
from keras import backend as K

import math

lambdas = {
    "const" :    lambda x: 0,
    "linear":    lambda x: x,
    "sin"   :    lambda x: math.sin(math.pi * x / 2),
    "sin+"  :    lambda x: math.sin(math.pi * x / 2),
    "sin-"  :    lambda x: (-1) * math.sin(math.pi * x / 2) + 1,
    "cos"   :    lambda x: math.cos(math.pi * x / 2),
    "cos-"  :    lambda x: math.cos(math.pi * x / 2),
    "cos+"  :    lambda x: (-1) * math.cos(math.pi * x / 2) + 1
}

class LRVariator(CallbackModule):
    """ This callback allows learning rate variations within or across epochs.
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
            fromVal: start value. If the param is omited, its value is taken from the keras model
            toVal: end value
            style: shape of the variation graphic. One of
              - linear
              - const
              - cos+ ascending cosine segment: -1 * cos(x * pi/2) + 1 for x in [0;1]
              - cos- descending cosine segment: cos(x * pi/2) for x in [0;1]
              - cos  same as 'cos-'
              - sin+ ascending sine segment: sin(x * pi/2) x in [0;1]
              - sin- descending sine segment: -1 * sin(x * pi/2) + 1 for x in [0;1]
              - sin  same as 'sin+'
              - any positive float or integer value 'a'. x^a for x in [0;1]
            args: see CallbackModule for details
        """
    
    extra_params=[{"name":"relSize","kind":"any","type":"number"},{"name":"absSize","kind":"any","type":"int","defaultValue":""},
                  {"name":"then","kind":"object","defaultValue":""},
                  {"name":"periodEpochs","kind":"number","defaultValue":""},
                  {"name":"periodSteps","kind":"int","defaultValue":""}
                  ]    
    
    def __init__(self, fromVal=None, toVal=0.006, style="linear", **args):
        super(LRVariator, self).__init__(**args)

        self.fromVal = fromVal
        self.toVal = toVal
        self.style = style
        self.lrComputer = None
        self.core = None

        if self.style not in lambdas:
            lambdasList = ", ".join([f"'{x}'" for x in lambdas])
            msg = f"LRVariator 'style' must be a positive number or one of {lambdasList}, but {self.style} have been recieved"
            if not isinstance(self.style,int) and not isinstance(self.style,float):
                raise ValueError(msg)
            elif self.style <= 0:
                raise ValueError(msg)


    def on_batch_end_action(self, logs = None):
        lr = self.get_lr()
        K.set_value(self.model.optimizer.lr, lr)
        print(f"    lr: {lr}")


    def get_lr(self):
        if self.lrComputer is None:
            if self.core is None:
                if self.fromVal is None:
                    self.fromVal = K.get_value(self.model.optimizer.lr)

                if self.style in lambdas:
                    self.core = lambdas[self.style]
                elif self.style == 1:
                    self.core = lambdas["linear"]
                elif isinstance(self.style, int):
                    pow = int(self.style)
                    self.core = lambda x: x ** pow
                elif isinstance(self.style, float):
                    pow = float(self.style)
                    self.core = lambda x: math.pow(x, pow)

            def lrComputerBody(x):
                point = (x - self.startStep) / (self.ownSteps - 1)
                ratio = self.core(point)
                dif = self.toVal - self.fromVal
                result = self.fromVal + ratio * dif
                result = max(result, 0.0)
                return result


            self.lrComputer = lambda x: lrComputerBody(x)

        lr = self.lrComputer(self.cyclicStep)
        return lr
