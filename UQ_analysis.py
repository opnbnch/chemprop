import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from sklearn.metrics import r2_score
from scipy import stats


def _check_col_type(df, col, df_type):
    """
    Each column we are working with should be of type "float".
    Checks to ensure this is true, otherwise raises an error.
    :Pandas.DataFrame df: df of interest
    :str col: column of interest
    :str df_type: df we are looking at
    """

    if df[col].dtype != 'float':
        raise ValueError(f'{df_type} column not found')


def _get_col(df, col, df_type):
    """
    Gets the column of interest in a csv. Uses the optional
    col arg if provided, otherwise will assume it is the
    second column in the df. Used for getting the value_col
    and the preds_col.
    :Pandas.DataFrame df: df of interest
    :str col: provided col arg
    """

    columns = df.columns
    if col:
        if col not in columns:
            raise ValueError(f'{df_type} column not found in test_csv')
    else:
        # expect column to be the second column in df
        col = columns[1]

    _check_col_type(df, col, df_type)

    return df[col]


def _get_uncertainty_col(df, uncertainty_col, split_UQ):
    """
    Gets the uncertainty column of the test_csv. Uses the optional
    value_col arg if provided, otherwise will assume it is the
    second column in the df. If split_UQ is flagged, will look for
    two columns--epistemic and aleatoric.
    :Pandas.DataFrame df: the preds_csv
    :str uncertainty_col: provided uncertainty_col arg
    :bool split_UQ: uncertainty is split to aleatoric, epistemic
    """

    columns = df.columns
    list_cols = list(columns)
    if uncertainty_col:
        if uncertainty_col not in columns:
            raise ValueError('Uncertainty column not found in test_csv')

        _check_col_type(df, uncertainty_col, 'Uncertainty')
        return df[uncertainty_col]
    else:
        if split_UQ:
            try:
                aleatoric_idx = list_cols.index('Aleatoric')
                epistemic_idx = list_cols.index('Epistemic')
            except ValueError:
                raise ValueError('Uncertainty appears not to be split')

            aleatoric_col = columns[aleatoric_idx]
            epistemic_col = columns[epistemic_idx]

            _check_col_type(df, aleatoric_col, 'Aleatoric')
            _check_col_type(df, epistemic_col, 'Epistemic')
            return aleatoric_col, epistemic_col
        else:
            uncertainty_idx = list_cols.index('Uncertainty')
            uncertainty_col = columns[uncertainty_idx]

            _check_col_type(df, uncertainty_col, 'Uncertainty')
            return df[uncertainty_col]


def _combine_cols(df, col_names):
    """
    Sums the aleatoric and epistemic columns to make a new
    uncertainty column for analysis.
    :Pandas.DataFrame df: preds dataframe
    :tuple col_names: strings of names of aleatoric, epistemic cols
    """

    # TODO: Allow analysis for both uncertainties?
    aleatoric, epistemic = col_names
    df['Uncertainty'] = df. \
        loc[:, [aleatoric, epistemic]].sum(axis=1)
    return df['Uncertainty']


def _get_r2(x, y, rounding=4):
    """
    Gets the r2 value between two columns. Rounds the value
    to the order provided (default 4).
    :pandas.DataSeries x: first series
    :pandas.DataSeries y: second series
    """

    sig_fig = '%.' + str(rounding) + 'g'
    r2 = r2_score(x, y)
    return float(sig_fig % r2)


def _create_scatter(x, y):
    """
    Create a scatter plot between two series.
    :Pandas.DataSeries x: first series
    :Pandas.DataSeries y: second series
    """
    plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    y_hat = np.poly1d(z)(x)

    plt.plot(x, y_hat, 'r--', lw=1)
    plt.xlabel('True Value', fontsize=16)
    plt.ylabel('Predicted Value', fontsize=16)
    plt.title('True vs. Predicted', fontsize=20)

    plt.show()


def _save_analysis(preds_r2, unc_r2, spearman, args):
    """
    Saves the values from analysis to a .json file.
    Uses the directory given and strips the name of the
    predictions file to name this analysis file.
    :float preds_r2: r2 value for predictions calculated
    :float unc_r2: r2 value for uncertaitny calculated
    :float spearman: spearman value calculated
    :args: args provided by user
    """

    preds_name = args.preds_path
    save_dir = args.save_dir

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    prefix = preds_name.split('/')[-1][:-4] + '_analysis.json'
    path = os.path.join(save_dir, prefix)

    data = {'Predictions R^2': preds_r2,
            'Uncertainty R^2': unc_r2,
            'Spearman': spearman}

    print('Saving analysis to ' + path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def analyze(args):
    test_csv = pd.read_csv(args.test_path)
    preds_csv = pd.read_csv(args.preds_path)
    value_col = _get_col(test_csv, args.value_col, 'Value')
    preds_col = _get_col(preds_csv, args.preds_col, 'Preds')
    uncertainty_col = _get_uncertainty_col(preds_csv, args.uncertainty_col,
                                           args.split_UQ)

    # Combine aleatoric and epistemic to make an uncertainty column
    if type(uncertainty_col) == tuple:
        uncertainty_col = _combine_cols(preds_csv, uncertainty_col)

    assert value_col.size == uncertainty_col.size == preds_col.size

    abs_err = (value_col - preds_col).abs()

    preds_r2 = _get_r2(value_col, preds_col)
    unc_r2 = _get_r2(abs_err, uncertainty_col)
    print('\nR2 for value <-> predictions: ' + str(preds_r2))
    print('R2 for absolute err <-> uncertainty: ' + str(unc_r2))

    spearman, _ = stats.spearmanr(abs_err, uncertainty_col)
    spearman = float('%.4g' % spearman)
    print('Spearman value for absolute err <-> uncertainty: ' + str(spearman))

    if args.save_dir:
        _save_analysis(preds_r2, unc_r2, spearman, args)

    if not args.quiet:
        _create_scatter(value_col, preds_col)


def analysis_outside(outside_args):
    """
    Used for calling this script from another python script.
    :dict outside_args: dict of args to use
    """

    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    args.test_path = outside_args['--test_path']
    args.preds_path = outside_args['--preds_path']
    args.save_dir = outside_args['--save_dir']
    args.quiet = True

    analyze(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('test_path', type=str,
                        help='Path to a csv file containing test data')
    parser.add_argument('preds_path', type=str,
                        help='Path to a csv file containing preds')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save analysis')
    parser.add_argument('--value_col', '--v', type=str, default=None,
                        help='Value column of test_csv')
    parser.add_argument('--uncertainty_col', '--UQ', type=str, default=None,
                        help='Uncertainty column of preds_csv')
    parser.add_argument('--preds_col', '--p', type=str, default=None,
                        help='Predictions column of preds_csv')
    parser.add_argument('--split_UQ', action='store_true', default=False,
                        help='Uncertainty is split to aleatoric, epistemic')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Supress graphs')
    args = parser.parse_args()

    analyze(args)
