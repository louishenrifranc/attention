from collections import namedtuple
import argparse
import json
import os

from attention.utils.config import AttrDict
from attention.algorithms import TransformerAlgorithm
from attention.algorithms.transformer.inputs_fn import create_textline_file

DatasetDirs = namedtuple("DatasetDirs", "train_data_dir valid_data_dir test_data_dir")


class TrainAttention(object):
    def __init__(self, config, train_data_dir, valid_data_dir, output_dir, metadata):
        self.output_dir = output_dir
        self.datasets = DatasetDirs(train_data_dir=train_data_dir,
                                    valid_data_dir=valid_data_dir,
                                    test_data_dir=None)
        self.output_dir = output_dir
        self.config = AttrDict(config)

        self._metadata = AttrDict(metadata)

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
        parser.add_argument('--valid_data_dir', help='Path to the valid dir', required=True)
        parser.add_argument('--output_dir', help='Path to the output dir', required=True)
        parser.add_argument('-c', '--config', help='Path to the configuration.json',
                            dest='configuration_file', required=True)
        parsed_args = parser.parse_args()
        return parsed_args

    def create_txt_filename(self, dataset_type):
        directory = getattr(self.datasets, dataset_type)

        def dialogue_generator():
            filenames = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]
            for filename in filenames:
                with open(filename, 'rt', encoding='utf-8') as f:
                    for dialogue in json.load(f):
                        yield dialogue

        txt_directory = os.path.join(self.output_dir, os.path.basename(directory))
        os.makedirs(txt_directory)
        context_filename = os.path.(txt_directory, "context.txt")
        answer_filename = os.path.join(txt_directory, "answer.txt")
        create_textline_file(dialogue_gen=dialogue_generator(),
                             context_filename=context_filename,
                             answer_filename=answer_filename)
        return context_filename, answer_filename

    def main(self):
        """Initializes a model and starts training using the args provided
        """

        params = self.config.model_params
        params["eos_token"] = self._metadata["eos_token"]
        params.encoder_params.embed_params["vocab_size"] = \
            params.decoder_params.embed_params["vocab_size"] = \
            params.decoder_params.params["vocab_size"] = self._metadata["vocab_size"]

        model = TransformerAlgorithm(estimator_run_config=params.estimator_params, params=params)
        self.train(model=model)

    def train(self, model):
        context_filename, answer_filename = self.create_txt_filename("train_data_dir")
        model.train(train_params=self.config.train_params,
                    train_answer_filename=answer_filename,
                    train_context_filename=context_filename)


if __name__ == '__main__':
    args = TrainAttention.parse_args()

    with open(args.config, mode='r') as config_file:
        args.config = json.load(config_file)

    with open(args.metadata, mode='r') as metadata_file:
        args.metadata = json.load(metadata_file)

    TrainAttention(**args)
