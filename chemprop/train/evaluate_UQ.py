import numpy as np
import tqdm

from .predict import predict

from chemprop.utils import get_avg_UQ


class Dropout_VI:
    """
    Uncertainty method for calculating aleatoric + epistemic UQ
    by using dropout and averaging over a number of predictions.
    """

    def __init__(self, args, data_loader, scaler):
        self.args = args
        self.scaler = scaler
        self.data_loader = data_loader
        self.split_UQ = args.split_UQ
        self.num_preds = args.num_preds

    def UQ_predict(self, model, sum_batch, sum_var, N):
        for i in tqdm.tqdm(range(self.num_preds)):
            batch_preds, var_preds = predict(
                                        model=model,
                                        data_loader=self.data_loader,
                                        disable_progress_bar=True,
                                        scaler=self.scaler
                                        )
            batch_preds = [item for sublist in batch_preds for item in sublist]
            sum_batch[:, i * N] = batch_preds
            sum_var[:, i * N] = var_preds
        return sum_batch, sum_var

    def calculate_UQ(self, sum_batch, sum_var):
        avg_preds = np.nanmean(sum_batch, 1).tolist()
        avg_UQ = get_avg_UQ(sum_var, avg_preds, sum_batch, return_both=self.split_UQ)

        return avg_preds, avg_UQ


class Ensemble_estimator:
    """
    Uncertainty method for calculating UQ by averaging
    over a number of ensembles of models.
    """

    def __init__(self, args, data_loader, scaler):
        self.args = args
        self.scaler = scaler
        self.data_loader = data_loader
        self.split_UQ = args.split_UQ


def uncertainty_estimator_builder(uncertainty_method):
    """
    Directory for getting the correct class of uncertainty method
    depending upon which one the user has specified.
    """
    return {
        'Dropout_VI': Dropout_VI,
        'Ensemble': Ensemble_estimator
    }[uncertainty_method]
