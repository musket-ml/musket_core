from callback_module import CallbackModule
from keras import backend as K
import math

lambdas = {
    "const" :    lambda x: 0,
    "linear":    lambda x: x,
    "sin"   :    lambda x: math.sin(2 * x / math.pi),
    "sin+"  :    lambda x: math.sin(2 * x / math.pi),
    "sin-"  :    lambda x: (-1) * math.sin(2 * x / math.pi) + 1,
    "cos"   :    lambda x: math.cos(2 * x / math.pi),
    "cos-"  :    lambda x: math.cos(2 * x / math.pi),
    "cos+"  :    lambda x: (-1) * math.cos(2 * x / math.pi) + 1
}

class LRVariator(CallbackModule):

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
