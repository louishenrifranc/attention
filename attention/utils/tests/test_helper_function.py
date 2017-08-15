from attention.utils.config import AttrDict
from attention.utils.helper_function import switch_dropout_off
import unittest


class TestHelperFunction(unittest.TestCase):
    def test_switch_off_dropout(self):
        old_dict = AttrDict({
            "dropout_rate": 10.0,
            "num_layers": 3,
            "model_params": {
                "dropout_rate": 10.0,
                "encoder_params": {
                    "num_layers": 3,
                    "dropout_rate": 10.0,
                    "encoder_block_params": {
                        "dropout_rate": 10.0,
                        "num_layers": 10
                    }
                }
            }
        })
        new_dict = switch_dropout_off(old_dict)
        self.assertEqual(new_dict, {'dropout_rate': 0.0, 'num_layers': 3,
                                    'model_params': {'dropout_rate': 0.0, 'encoder_params': {
                                        'encoder_block_params': {'dropout_rate': 0.0, 'num_layers': 10},
                                        'dropout_rate': 0.0, 'num_layers': 3}}})
