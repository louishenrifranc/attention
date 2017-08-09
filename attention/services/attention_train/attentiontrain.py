from collections import namedtuple
import argparse
import json
import os

from attention.utils.config import AttrDict, RunConfig
from attention.algorithms import TransformerAlgorithm
from attention.algorithms.transformer.inputs_fn import create_textline_file

DatasetDirs = namedtuple("DatasetDirs", "train_data_dir valid_data_dir test_data_dir")


class TrainAttention(object):
    def __init__(self, config, train_data_dir, valid_data_dir, output_dir, metadata):
        self.output_dir = output_dir
        self.datasets = DatasetDirs(train_data_dir=train_data_dir,
                                    valid_data_dir=valid_data_dir,
                                    test_data_dir=None)
        self.config = AttrDict.from_nested_dict(config)

        self._metadata = AttrDict.from_nested_dict(metadata)

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
        parser.add_argument('--metadata', help='Path to the metadata', required=True)
        parser.add_argument('--valid_data_dir', help='Path to the valid dir')
        parser.add_argument('--output_dir', help='Path to the output dir', required=True)
        parser.add_argument('-c', '--config', help='Path to the configuration.json', required=True)
        parsed_args = parser.parse_args()
        return parsed_args


    def main(self):
        """Initializes a model and starts training using the args provided
        """

        params = self.config.model_params
        params.pad_token = self._metadata.pad_token
        params.encoder_params.embed_params.vocab_size = \
            params.decoder_params.embed_params.vocab_size = \
            params.decoder_params.params.vocab_size = self._metadata.vocab_size
        estimator_params = self.config.estimator_params
        estimator_params.model_dir = self.output_dir
        estimator_run_config = RunConfig().replace(**estimator_params)
        model = TransformerAlgorithm(estimator_run_config=estimator_run_config, params=params)

        if self.datasets.valid_data_dir is not None:
            self.train_and_evaluate(model=model)
        else:
            self.train(model=model)

    def train(self, model):
        model.train(train_params=self.config.train_params,
                                 train_context_filename=os.path.join(self.datasets.train_data_dir, "context.txt"),
                                 train_answer_filename=os.path.join(self.datasets.train_data_dir, "answer.txt"))

    def train_and_evaluate(self, model):
        model.train_and_evaluate(train_params=self.config.train_params,
                                 train_context_filename=os.path.join(self.datasets.train_data_dir, "context.txt"),
                                 train_answer_filename=os.path.join(self.datasets.train_data_dir, "answer.txt"),
                                 validation_params=self.config.train_params,
                                 validation_context_filename=os.path.join(self.datasets.valid_data_dir, "context.txt"),
                                 validation_answer_filename=os.path.join(self.datasets.valid_data_dir, "answer.txt"))


if __name__ == '__main__':
    args = TrainAttention.parse_args()

    with open(args.config, mode='r') as config_file:
        args.config = json.load(config_file)

    with open(args.metadata, mode='r') as metadata_file:
        args.metadata = json.load(metadata_file)
    args = vars(args)
    TrainAttention(**args).main()
