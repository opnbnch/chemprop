from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A MoleculeDataLoader.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """

    UQ = model.uncertainty
    two_vals = UQ == 'Dropout_VI' or UQ == 'Ensemble'
    mve = model.mve
    training = model.training
    if UQ != 'Dropout_VI':
        model.eval()

    total_batch_preds = []
    total_var_preds = []
    for batch in tqdm(data_loader, disable=disable_progress_bar):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        # Make predictions
        if two_vals:
            with torch.no_grad():
                batch_preds, logvar_preds = model(mol_batch, features_batch)
            var_preds = torch.exp(logvar_preds)
            var_preds = var_preds.data.cpu().numpy().tolist()
            total_var_preds.extend(var_preds)

        else:
            with torch.no_grad():
                batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        total_batch_preds.extend(batch_preds)

    if mve:
        p = []
        c = []
        for i in range(len(total_batch_preds)):
            p.append([total_batch_preds[i][j] for j in range(len(total_batch_preds[i])) if j % 2 == 0])
            c.append([total_batch_preds[i][j] for j in range(len(total_batch_preds[i])) if j % 2 == 1])

        if scaler is not None:
            p = scaler.inverse_transform(p).tolist()
            c = (scaler.stds**2 * c).tolist()

        if not training:
            return p, c
        else:
            return p

    # Inverse scale if regression
    if scaler is not None:
        total_batch_preds = scaler.inverse_transform(total_batch_preds).tolist()

    if not UQ or training or not two_vals:
        return total_batch_preds
    else:
        return total_batch_preds, total_var_preds
