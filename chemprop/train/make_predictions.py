import csv
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .predict import predict
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs
from .evaluate_UQ import uncertainty_estimator_builder


def make_predictions(args: PredictArgs, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    # If features were used during training, they must be used when predicting
    if ((train_args.features_path is not None or train_args.features_generator is not None)
            and args.features_path is None
            and args.features_generator is None):
        raise ValueError('Features were used during training so they must be specified again during prediction '
                         'using the same type of features as before (with either --features_generator or '
                         '--features_path and using --no_features_scaling if applicable).')

    # Update predict args with training arguments to create a merged args object
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args: Union[PredictArgs, TrainArgs]

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, features_generator=args.features_generator)
    else:
        full_data = get_data(path=args.test_path, args=args, target_columns=[], skip_invalid_smiles=False)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if full_data[full_index].mol is not None:
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if not args.uncertainty:
        if args.dataset_type == 'multiclass':
            sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
        else:
            sum_preds = np.zeros((len(test_data), num_tasks))
    else:
        if args.uncertainty == 'Dropout_VI':
            sum_batch = np.zeros((len(test_data), len(args.checkpoint_paths) * args.num_preds))
            sum_var = np.zeros((len(test_data), len(args.checkpoint_paths) * args.num_preds))
        else:
            sum_batch = np.zeros((len(test_data), len(args.checkpoint_paths)))
            sum_var = np.zeros((len(test_data), len(args.checkpoint_paths)))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    if args.uncertainty:
        uncertainty_estimator = uncertainty_estimator_builder(args.uncertainty)(args, test_data_loader, scaler)

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for N, checkpoint_path in tqdm(enumerate(args.checkpoint_paths), total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, device=args.device)
        model.training = False
        if not args.uncertainty:
            model_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            sum_preds += np.array(model_preds)
        else:
            sum_batch, sum_var = uncertainty_estimator.UQ_predict(model, sum_batch, sum_var, N * args.num_preds)

    # Ensemble predictions
    if not args.uncertainty:
        avg_preds = sum_preds / len(args.checkpoint_paths)
        avg_preds = avg_preds.tolist()
    else:
        avg_preds, avg_UQ = uncertainty_estimator.calculate_UQ(sum_batch, sum_var)
        if type(avg_UQ) is tuple:
            aleatoric = avg_UQ[0]
            epistemic = avg_UQ[1]

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    makedirs(args.preds_path, isfile=True)

    # Get prediction column names
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
    else:
        task_names = task_names

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)

        if args.uncertainty:
            if not args.split_UQ:
                cur_UQ = avg_UQ[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
                datapoint.row['Uncertainty'] = cur_UQ
            elif args.split_UQ:
                cur_al = aleatoric[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
                cur_ep = epistemic[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
                datapoint.row['Aleatoric'] = cur_al
                datapoint.row['Epistemic'] = cur_ep

        if type(preds) is list:
            for pred_name, pred in zip(task_names, preds):
                datapoint.row[pred_name] = pred
        else:
            datapoint.row[task_names[0]] = preds


    # Save
    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
        writer.writeheader()

        for datapoint in full_data:
            writer.writerow(datapoint.row)

    return avg_preds
