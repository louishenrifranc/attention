from config import Config

import tensorflow as tf
import sonnet as snt


class PointWiseFeedForward(snt.AbstractModule):
    def __init__(self):
        super(PointWiseFeedForward, self).__init__(name="pointwise_feedforward")
        pass

    def _build(self, inputs):
        pass
