import numpy as np
import argparse
import json
import shutil
import os

from attention.utils.config import AttrDict


class CreateCopyTask(object):
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.config = AttrDict(config)

    @staticmethod
    def parse_args():
        """ Parses the arguments through the ArgumentParser
        Args:
            args (list): list of arguments to be parsed (should be sys.argv[1:])
        Returns:
            argsparse object
        """
        parser = argparse.ArgumentParser(description="Creating CopyTask for Transformer")
        parser.add_argument('--output_dir', help='Path to the output dir', required=True)
        parser.add_argument('-c', '--config', help='Path to the configuration.json',
                            required=True)
        parsed_args = parser.parse_args()
        return parsed_args

    def create_copy_task_files(self, context_filename, answer_filename, vocab_size, num_examples, max_sequence_length):
        with open(context_filename, 'w') as file:
            for _ in range(num_examples):
                num_tokens = np.random.randint(2, max_sequence_length, 1)
                tokens = np.random.randint(0, vocab_size, num_tokens)
                file.write(" ".join([str(x) for x in list(tokens)]) + "\n")

        shutil.copyfile(context_filename, answer_filename)

    def main(self):
        """Initializes a model and starts training using the args provided
        """
        train_dir = os.path.join(self.output_dir, "train")
        eval_dir = os.path.join(self.output_dir, "eval")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        self.create_copy_task_files(
            context_filename=os.path.join(train_dir, "context.txt"),
            answer_filename=os.path.join(train_dir, "answer.txt"),
            **self.config.train_params
        )

        self.create_copy_task_files(
            context_filename=os.path.join(eval_dir, "context.txt"),
            answer_filename=os.path.join(eval_dir, "answer.txt"),
            **self.config.eval_params
        )


if __name__ == '__main__':
    args = CreateCopyTask.parse_args()

    with open(args.config, mode='r') as config_file:
        args.config = json.load(config_file)
    args = vars(args)
    CreateCopyTask(**args).main()
