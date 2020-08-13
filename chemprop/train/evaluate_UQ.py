import numpy as np
import tqdm

from .predict import predict

from chemprop.utils import get_avg_UQ


class Uncertainty_estimator:
    """
    General class with methods for UQ.
    """

    def __init__(self, args, scaler):
        self.args = args
        self.scaler = scaler
        self.split_UQ = args.split_UQ


class Dropout_VI(Uncertainty_estimator):
    """
    Uncertainty method for calculating aleatoric + epistemic UQ
    by using dropout and averaging over a number of predictions.
    """

    def __init__(self, args, scaler):
        super().__init__(args, scaler)
        self.num_preds = args.num_preds

    def UQ_predict(self, model, sum_batch, sum_var, data_loader, N=0):
        offset = N * self.num_preds

        for i in tqdm.tqdm(range(self.num_preds)):
            batch_preds, var_preds = predict(
                                        model=model,
                                        data_loader=data_loader,
                                        disable_progress_bar=True,
                                        scaler=self.scaler
                                        )
            # Ensure they are in proper list form
            batch_preds = [item for sublist in batch_preds for item in sublist]
            var_preds = [item for sublist in var_preds for item in sublist]
            sum_batch[:, i + offset] = batch_preds
            sum_var[:, i + offset] = var_preds
        return sum_batch, sum_var

    def calculate_UQ(self, sum_batch, sum_var):
        avg_preds = np.nanmean(sum_batch, 1).tolist()
        avg_UQ = get_avg_UQ(sum_var, avg_preds, sum_batch, return_both=self.split_UQ)

        return avg_preds, avg_UQ


class Ensemble_estimator(Uncertainty_estimator):
    """
    Uncertainty method for calculating UQ by averaging
    over a number of ensembles of models.
    """

    def __init__(self, args, scaler):
        super().__init__(args, scaler)

    def UQ_predict(self, model, sum_batch, sum_var, data_loader, N=0):
        batch_preds, var_preds = predict(
                            model=model,
                            data_loader=data_loader,
                            disable_progress_bar=True,
                            scaler=self.scaler
                            )

        # Ensure they are in proper list form
        batch_preds = [item for sublist in batch_preds for item in sublist]
        var_preds = [item for sublist in var_preds for item in sublist]
        sum_batch[:, N] = batch_preds
        sum_var[:, N] = var_preds

        return sum_batch, sum_var

    def calculate_UQ(self, sum_batch, sum_var):
        breakpoint()
        avg_preds = np.nanmean(sum_batch, 1).tolist()

        aleatoric = np.nanmean(sum_var, 1).tolist()
        epistemic = np.var(sum_batch, 1).tolist()

        total_unc = aleatoric + epistemic

        if self.split_UQ:
            return avg_preds, (aleatoric, epistemic)
        else:
            return avg_preds, total_unc


def uncertainty_estimator_builder(uncertainty_method):
    """
    Directory for getting the correct class of uncertainty method
    depending upon which one the user has specified.
    """
    return {
        'Dropout_VI': Dropout_VI,
        'Ensemble': Ensemble_estimator
    }[uncertainty_method]
