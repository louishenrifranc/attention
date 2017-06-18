from config import Config

import tensorflow as tf
import sonnet as snt


class Transformer(snt.AbstractModule):
    def __init__(self):
        super(Transformer, self).__init(name="transformer")
        pass

    def _build(self):
        pass
