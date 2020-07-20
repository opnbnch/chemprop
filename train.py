"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger
from chemprop.process_args_file import get_args_list
import sys


if __name__ == '__main__':
    # Process args from a file
    # python train.py --data_path path args_file.json
    line_args, file_name = get_args_list(sys.argv, len(sys.argv) - 1)
    sys.argv = sys.argv + line_args
    sys.argv.remove(file_name)

    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
