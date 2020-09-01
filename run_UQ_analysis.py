from train import train_outside
from predict import predict_outside
from UQ_analysis import analysis_outside
import json
import argparse
import os
from tqdm import tqdm

UQ_methods = ['Dropout_VI', 'Ensemble', 'random_forest', 'gaussian', 'mve']


def load_file(file_path):
    """
    Loads and returns a json file.
    :str file_path: path to file to load
    """

    with open(file_path) as f:
        cur_file = json.load(f)

    return cur_file


def append_train_args(base_args, data_name, path, uncertainty):
    """
    Appends dataset and UQ specific args to the base train args for training.
    :dict base_args: basic training args
    :str data_name: name of dataset currently being used
    :str path: path of dataset currently being used
    :str uncertainty: UQ method being used
    """
    save_dir = 'ckpnts/' + uncertainty + '_' + data_name[:-4]
    features_path = 'features/' + data_name[:-4] + '_features.npz'

    additional_args = {'--data_path': path,
                       '--save_dir': save_dir,
                       '--uncertainty': uncertainty,
                       '--features_path': features_path}
    base_copy = base_args.copy()
    base_copy.update(additional_args)

    return base_copy


def append_predict_args(base_args, ckpnt_path, test_path, preds_dir):
    """
    Appends dataset and UQ specific args to the base predict args.
    :dict base_args: basic predict args
    :str ckpnt_path: current ckpnt loc
    :str test_path: path to test csv to use, None by default
    :str preds_dir: directory to save preds, None by default
    """

    if not test_path:
        test_path = os.path.join(ckpnt_path, 'fold_0', 'test_full.csv')

    preds_file_name = ckpnt_path.split('/')[-1]
    directory = preds_dir if preds_dir is not None else 'predictions'
    preds_path = os.path.join(directory, preds_file_name + '_preds.csv')

    additional_args = {'--test_path': test_path,
                       '--preds_path': preds_path,
                       '--checkpoint_dir': ckpnt_path}

    base_copy = base_args.copy()
    base_copy.update(additional_args)

    return base_copy


def main(args_dir, data_dir, test_path, preds_dir):
    """
    Main function to control reading in args files, appending new
    args to them, and training, predicting, and analyzing data.
    :str args_dir: directory to args files
    :str data_dir: directory that holds all data to anaylze
    :str test_path: optional path to test csv to use
    :str preds_dir: optional directory to hold predictions
    """

    # Expect file names in args_dir to be of this format
    train_args_path = os.path.join(args_dir, 'train_args.json')
    predict_args_path = os.path.join(args_dir, 'predict_args.json')
    base_train_args = load_file(train_args_path)
    base_predict_args = load_file(predict_args_path)

    dataset_list = [f for f in os.listdir(data_dir) if not f.startswith('.')]

    for num, dataset in tqdm(enumerate(dataset_list)):
        print('Current dataset: ' + dataset)
        path = os.path.join(data_dir, dataset)

        for uncertainty in tqdm(UQ_methods):
            print('Current uncertainty: ' + uncertainty)
            # Training
            train_args_dict = append_train_args(base_train_args, dataset,
                                                path, uncertainty)
            train_outside(train_args_dict)

            # Predicting
            pred_args_dict = append_predict_args(base_predict_args,
                                                 train_args_dict['--save_dir'],
                                                 test_path,
                                                 preds_dir)
            predict_outside(pred_args_dict)

            # Analysis
            analysis_path = os.path.join('analysis', dataset[:-4])
            analysis_save = {'--save_dir': analysis_path}

            pred_args_dict.update(analysis_save)
            analysis_outside(pred_args_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('args_dir', type=str,
                        help='Path to directory with base args')
    parser.add_argument('data_dir', type=str,
                        help='Path to directory with datasets to use')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to dataset to use for predictions')
    parser.add_argument('--preds_dir', type=str, default=None,
                        help='Path to directory to save predictions')
    args = parser.parse_args()

    main(args.args_dir, args.data_dir, args.test_path, args.preds_dir)
