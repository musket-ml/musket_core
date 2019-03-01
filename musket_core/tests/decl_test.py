import unittest
from musket_core import net_declaration
import keras


class TestStringMethods(unittest.TestCase):

    def testNetCreation(self):
        m1=net_declaration.create_model("../examples/example2.9.yaml",keras.layers.Input((200,200,3)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.8.yaml",keras.layers.Input((200,200,3)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.7.yaml",keras.layers.Input((200,200,3)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.6.yaml",keras.layers.Input((200,200,3)))
        print(m1.summary())

        m1=net_declaration.create_model("../examples/example2.5.yaml",keras.layers.Input((200,200,3)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.4.yaml",keras.layers.Input((200,200,3)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.3.yaml",keras.layers.Input((200,200)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.2.yaml",keras.layers.Input((200,200)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example2.1.yaml",keras.layers.Input((200,200)))
        print(m1.summary())
        m1=net_declaration.create_model("../examples/example1.yaml",keras.layers.Input((200,200)))
        print(m1.summary())
        m2=net_declaration.create_model("../examples/example2.yaml",[keras.layers.Input((200,200)),keras.layers.Input((200,200))])
        print(m2.summary())
        m3=net_declaration.create_model("../examples/example3.yaml",[keras.layers.Input((200,200)),keras.layers.Input((200,200))])
        assert len(m3.outputs)==2
        print(m3.summary())
        m4=net_declaration.create_model("../examples/inception.yaml",[keras.layers.Input((200,200)),keras.layers.Input((200,200))])
        print(m4.summary())
        m5=net_declaration.create_model("../examples/simple.yaml",[keras.layers.Input((200,200)),keras.layers.Input((200,200))])
        print(m5.summary())
        m6 = net_declaration.create_model("../examples/bidirectional.yaml",
                                          [keras.layers.Input((200, 200))])
        print(m6.summary())
