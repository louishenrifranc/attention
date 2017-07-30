from collections import namedtuple
from types import SimpleNamespace
import argparse
import json

from algorithms import TransformerAlgorithm

DatasetDirs = namedtuple("DatasetDirs", "train_data_dir valid_data_dir test_data_dir")


class TrainAttention(object):
    def __init__(self, config, train_data_dir, valid_data_dir, output_dir):
        self.output_dir = output_dir
        self.datasets = DatasetDirs(train_data_dir=train_data_dir,
                                    valid_data_dir=valid_data_dir,
                                    test_data_dir=None)
        self.output_dir = output_dir
        self.config = config

    @staticmethod
    def parse_args():
        """ Parses the arguments through the ArgumentParser
        Args:
            args (list): list of arguments to be parsed (should be sys.argv[1:])
        Returns:
            argsparse object
        """
        parser = argparse.ArgumentParser(description="Training Program for Deep Learning Models")
        parser.add_argument('--train_data_dir', help='Path to the train dir', required=True)
        parser.add_argument('--valid_data_dir', help='Path to the valid dir', required=True)
        parser.add_argument('--output_dir', help='Path to the output dir', required=True)
        parser.add_argument('-c', '--config', help='Path to the configuration.json',
                            dest='configuration_file', required=True)
        parsed_args = parser.parse_args()
        return parsed_args

    def main(self):
        """Initializes a model and starts training using the args provided
        """

        params = self.config.model_params.to_dict()
        model = TransformerAlgorithm(estimator_config=None, params=params)
        self.train(model=model)

    def train(self, model):
        model.train(train_params=self.config.train.input_params.to_dict())


if __name__ == '__main__':
    args = TrainAttention.parse_args()

    with open(args.config, mode='r') as config_file:
        args.config = json.load(config_file, object_hook=lambda d: SimpleNamespace(**d))
    TrainAttention(**args)
