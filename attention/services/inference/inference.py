from attention.utils.helper_function import switch_dropout_off
from attention.utils.config import AttrDict, RunConfig
from attention.algorithms import TransformerAlgorithm

import numpy as np
import argparse
import json
import os


class InferAttention:
    def __init__(self, predict_file, config, model_dir, output_dir, metadata):
        self.predict_file = predict_file
        self.model_dir = model_dir

        self.config = AttrDict.from_nested_dict(config)
        self._metadata = AttrDict.from_nested_dict(metadata)

        self.output_dir = output_dir

    @staticmethod
    def parse_args():
        """ Parses the arguments through the ArgumentParser
        Args:
            args (list): list of arguments to be parsed (should be sys.argv[1:])
        Returns:
            argsparse object
        """
        parser = argparse.ArgumentParser(description="Inference Program for Deep Learning Models")
        parser.add_argument('--predict_file', help='Path to the predict file', required=True)
        parser.add_argument('--metadata', help='Path to the metadata', required=True)
        parser.add_argument('--output_dir', help='Path to the output dir', required=True)
        parser.add_argument('-c', '--config', help='Path to the configuration.json', required=True)
        parsed_args = parser.parse_args()
        return parsed_args

    def main(self):
        """Initializes a model and run prediction
        """
        params = self.config.model_params
        params = switch_dropout_off(params)

        for name, value in self._metadata.special_tokens.items():
            params.name = value

        params.encoder_params.embed_params.vocab_size = \
            params.decoder_params.embed_params.vocab_size = \
            params.decoder_params.params.vocab_size = self._metadata.vocab_size

        estimator_params = self.config.estimator_params
        estimator_params.model_dir = self.model_dir
        estimator_run_config = RunConfig().replace(**estimator_params)

        model = TransformerAlgorithm(estimator_run_config=estimator_run_config, params=params)
        self.predict(model)

    def parse_input(self):
        with open(self.predict_file, "r") as f:
            content = f.readlines()
        content[:] = [np.array([int(x) for x in line.split(",")]) for line in content]
        try:
            inputs = np.array(content)
        except:
            raise ValueError("Inputs must have the same shape inside the prediction file")

        return inputs

    def predict(self, model):
        predictions = model.predict(inputs=self.parse_input(), predict_params=None)

        output_file = os.path.join(self.output_dir, "results")
        with open(output_file, "w") as f:
            f.write(predictions)


if __name__ == '__main__':
    args = InferAttention.parse_args()

    with open(args.config, mode='r') as config_file:
        args.config = json.load(config_file)

    with open(args.metadata, mode='r') as metadata_file:
        args.metadata = json.load(metadata_file)

    args = vars(args)
    InferAttention(**args).main()
