from callback_module import CallbackModule
from keras import backend as K
import math

class LRVariator(CallbackModule):

    def __init__(self, fromVal=None, toVal=0.006, style="linear", **args):
        super(LRVariator, self).__init__(**args)

        self.fromVal = fromVal
        self.toVal = toVal
        self.style = style
        self.lrComputer = None
        self.core = None

        if self.style != "linear":
            if not isinstance(self.style,int) and not isinstance(self.style,float):
                raise ValueError(f"LRVariator 'style' must be a positive number or 'linear', but {self.style} have been recieved")
            elif self.style <= 0:
                raise ValueError(f"LRVariator 'style' must be a positive number or 'linear', but {self.style} have been recieved")


    def on_batch_end_action(self, batch, logs = None):
        lr = self.get_lr()
        K.set_value(self.model.optimizer.lr, lr)
        print(f"    lr: {lr}")


    def get_lr(self):
        if self.lrComputer is None:
            if self.core is None:
                if self.fromVal is None:
                    self.fromVal = K.get_value(self.model.optimizer.lr)

                if self.style == "linear" or self.style == 1:
                    self.core = lambda x: x
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
                return result


            self.lrComputer = lambda x: lrComputerBody(x)

        lr = self.lrComputer(self.step)
        return lr
