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

    additional_args = {'--data_path': path,
                       '--save_dir': save_dir,
                       '--uncertainty': uncertainty}
    base_copy = base_args.copy()
    base_copy.update(additional_args)

    return base_copy


def append_predict_args(base_args, ckpnt_path):
    """
    Appends dataset and UQ specific args to the base predict args.
    :dict base_args: basic predict args
    :str ckpnt_path: current ckpnt loc
    """

    test_path = os.path.join(ckpnt_path, 'fold_0', 'test_full.csv')
    preds_file_name = ckpnt_path.split('/')[-1]
    preds_path = os.path.join('predictions', preds_file_name + '_preds.csv')

    additional_args = {'--test_path': test_path,
                       '--preds_path': preds_path,
                       '--checkpoint_dir': ckpnt_path}

    base_copy = base_args.copy()
    base_copy.update(additional_args)

    return base_copy


def main(train_args_path, pred_args_path, data_dir):

    base_train_args = load_file(train_args_path)
    base_predict_args = load_file(pred_args_path)

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
                                                 train_args_dict['--save_dir'])
            predict_outside(pred_args_dict)

            # Analysis
            analysis_path = os.path.join('analysis', dataset[:-4])
            analysis_save = {'--save_dir': analysis_path}

            pred_args_dict.update(analysis_save)
            analysis_outside(pred_args_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_args_path', type=str,
                        help='Path to a json file with basic train args')
    parser.add_argument('pred_args_path', type=str,
                        help='Path to a json file with basic predict args')
    parser.add_argument('data_dir', type=str,
                        help='Path to directory with datasets to use')
    args = parser.parse_args()

    main(args.train_args_path, args.pred_args_path, args.data_dir)
