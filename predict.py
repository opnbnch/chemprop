"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.args import PredictArgs
from chemprop.train import make_predictions
from chemprop.process_args_file import get_args_list
import sys

if __name__ == '__main__':
    # Process args from a file
    # python predict.py --test_path test --preds_path pred args_file.json
    sys.argv = get_args_list(sys.argv, len(sys.argv) - 1)

    args = PredictArgs().parse_args()
    make_predictions(args)
