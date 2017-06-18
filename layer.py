from config import Config

import tensorflow as tf
import sonnet as snt


class Layer(snt.AbstractModule):
    def __init__(self):
        super(Layer, self).__init(name="layer")
        pass

    def _build(self):
        pass
