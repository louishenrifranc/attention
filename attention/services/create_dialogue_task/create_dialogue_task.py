import argparse
import json
import shutil
import os

from attention.utils.config import AttrDict
from attention.algorithms.transformer.inputs_fn import create_textline_file

DatasetDirs = namedtuple("DatasetDirs", "train_data_dir valid_data_dir test_data_dir")


class CreateDialogueTask(object):
    def __init__(self, config, train_data_dir, valid_data_dir, output_dir):
        self.output_dir = output_dir
        self.datasets = DatasetDirs(train_data_dir=train_data_dir,
                                    valid_data_dir=valid_data_dir,
                                    test_data_dir=None)
        self.config = AttrDict(config)

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
        train_dir = os.path.join(self.output_dir, "train")
        eval_dir = os.path.join(self.output_dir, "eval")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        self.create_txt_filename(dataset_type="train")
        self.create_txt_filename(dataset_type="eval")


if __name__ == '__main__':
    args = CreateDialogueTask.parse_args()

    with open(args.config, mode='r') as config_file:
        args.config = json.load(config_file)

    # TODO: To finish
    CreateDialogueTask(**args).main()
