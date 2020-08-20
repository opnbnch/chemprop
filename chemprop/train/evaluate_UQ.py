import numpy as np
import GPy
import tqdm

from .predict import predict

from chemprop.utils import get_avg_UQ
from argparse import Namespace
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Callable, List, Tuple
from chemprop.data import MoleculeDataset, MoleculeDataLoader, StandardScaler
import torch.nn as nn


class UncertaintyEstimator:
    """
    An UncertaintyEstimator calculates uncertainty when passed a model.
    Certain UncertaintyEstimators also augment the model and alter prediction
    values. Note that many UncertaintyEstimators compute unscaled uncertainty
    values. These are only meaningful relative to one another.
    """
    def __init__(self,
                 args: Namespace,
                 data,
                 scaler: StandardScaler,
                 ):
        """
        Constructs an UncertaintyEstimator.
        :param train_data: The data a model was trained on.
        :param val_data: The validation/supplementary data for a model.
        :param test_data: The data to test the model with.
        :param scaler: A scaler the model uses to transform input data.
        :param args: The command line arguments.
        """

        self.args = args
        self.scaler = scaler
        self.data = data
        self.data_loader = MoleculeDataLoader(
                            dataset=data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers
        )
        self.data_len = len(data)
        self.data_width = len(args.checkpoint_paths)
        self.split_UQ = args.split_UQ

    def process_model(self, model: nn.Module, N=0):
        """Perform initialization using model and prior data.
        :param model: The model to learn the uncertainty of.
        :int N: Enumerate of model we are currently processing.
        """
        pass

    def calculate_UQ(self,
                     val_predictions: np.ndarray,
                     test_predictions: np.ndarray):
        """
        Compute uncertainty on self.val_data and self.test_data predictions.
        :param val_predictions: The predictions made on self.val_data.
        :param test_predictions: The predictions made on self.test_data.
        :return: Validation set predictions, validation set uncertainty,
                 test set predictions, and test set uncertainty.
        """
        pass

    def _scale_uncertainty(self, uncertainty: float) -> float:
        """
        Rescale uncertainty estimates to account for scaled input.
        :param uncertainty: An unscaled uncertainty estimate.
        :return: A scaled uncertainty estimate.
        """
        return self.scaler.stds * uncertainty


class Dropout_VI(UncertaintyEstimator):
    """
    Uncertainty method for calculating aleatoric + epistemic UQ
    by using dropout and averaging over a number of predictions.
    """

    def __init__(self, args, data, scaler):
        super().__init__(args, data, scaler)
        self.num_preds = args.num_preds
        self._create_matrix()

    def _create_matrix(self):
        self.data_width *= self.num_preds
        self.sum_batch = np.zeros((self.data_len, self.data_width))
        self.sum_var = np.zeros((self.data_len, self.data_width))

    def process_model(self, model, N=0):
        offset = N * self.num_preds

        for i in tqdm.tqdm(range(self.num_preds)):
            batch_preds, var_preds = predict(
                                        model=model,
                                        data_loader=self.data_loader,
                                        disable_progress_bar=True,
                                        scaler=self.scaler
                                        )
            # Ensure they are in proper list form
            # TODO: Allow for multitasking instead 
            batch_preds = [item for sublist in batch_preds for item in sublist]
            var_preds = [item for sublist in var_preds for item in sublist]
            self.sum_batch[:, i + offset] = batch_preds
            self.sum_var[:, i + offset] = var_preds

    def calculate_UQ(self):
        avg_preds = np.nanmean(self.sum_batch, 1).tolist()
        avg_UQ = get_avg_UQ(self.sum_var, avg_preds, self.sum_batch, return_both=self.split_UQ)

        return avg_preds, avg_UQ


class Ensemble_estimator(UncertaintyEstimator):
    """
    Uncertainty method for calculating UQ by averaging
    over a number of ensembles of models.
    """

    def __init__(self, args, data, scaler):
        super().__init__(args, data, scaler)
        self._create_matrix()

    def _create_matrix(self):
        self.sum_batch = np.zeros((self.data_len, self.data_width))
        self.sum_var = np.zeros((self.data_len, self.data_width))

    def process_model(self, model, N=0):
        batch_preds, var_preds = predict(
                            model=model,
                            data_loader=self.data_loader,
                            disable_progress_bar=True,
                            scaler=self.scaler
                            )

        # Ensure they are in proper list form
        batch_preds = [item for sublist in batch_preds for item in sublist]
        var_preds = [item for sublist in var_preds for item in sublist]
        self.sum_batch[:, N] = batch_preds
        self.sum_var[:, N] = var_preds

    def calculate_UQ(self):
        avg_preds = np.nanmean(self.sum_batch, 1).tolist()

        aleatoric = np.nanmean(self.sum_var, 1).tolist()
        epistemic = np.var(self.sum_batch, 1).tolist()

        total_unc = aleatoric + epistemic

        if self.split_UQ:
            return avg_preds, (aleatoric, epistemic)
        else:
            return avg_preds, total_unc


class ExposureEstimator(UncertaintyEstimator):
    """
    An ExposureEstimator drops the output layer
    of the provided model after training.
    The "exposed" final hidden-layer is used to calculate uncertainty.
    """
    def __init__(self,
                 args: Namespace,
                 data,
                 scaler: StandardScaler):
        super().__init__(args, data, scaler)

        self.sum_last_hidden_test = np.zeros(
            (len(self.data), self.args.last_hidden_size))

    def process_model(self, model: nn.Module, data_loader):
        model.eval()
        model.use_last_hidden = False

        last_hidden_test = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_test += np.array(last_hidden_test)

    def _compute_hidden_vals(self):
        ensemble_size = self.args.ensemble_size
        avg_last_hidden_train = self.sum_last_hidden_train / ensemble_size
        avg_last_hidden_val = self.sum_last_hidden_val / ensemble_size
        avg_last_hidden_test = self.sum_last_hidden_test / ensemble_size

        return avg_last_hidden_train, avg_last_hidden_val, avg_last_hidden_test


class RandomForestEstimator(ExposureEstimator):

    def calculate_UQ(self,
                     val_predictions: np.ndarray,
                     test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        A RandomForestEstimator trains a random forest to
        operate on data transformed by the provided model.
        Predictions are calculated using the output of the random forest.
        Reported uncertainty is the variance of trees in the forest.
        """
        (_,
         avg_last_hidden_val,
         avg_last_hidden_test) = self._compute_hidden_vals()

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        n_trees = 128
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(avg_last_hidden_val, transformed_val[:, task])

            avg_val_preds = forest.predict(avg_last_hidden_val)
            val_predictions[:, task] = avg_val_preds

            individual_val_predictions = np.array([estimator.predict(
                avg_last_hidden_val) for estimator in forest.estimators_])
            val_uncertainty[:, task] = np.std(individual_val_predictions,
                                              axis=0)

            avg_test_preds = forest.predict(avg_last_hidden_test)
            test_predictions[:, task] = avg_test_preds

            individual_test_predictions = np.array([estimator.predict(
                avg_last_hidden_test) for estimator in forest.estimators_])
            test_uncertainty[:, task] = np.std(individual_test_predictions,
                                               axis=0)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_uncertainty(val_uncertainty),
                test_predictions, self._scale_uncertainty(test_uncertainty))


class GaussianProcessEstimator(ExposureEstimator):
    """
    A GaussianProcessEstimator trains a Gaussian process to
    operate on data transformed by the provided model.
    Uncertainty and predictions are calculated using
    the output of the Gaussian process.
    """
    def calculate_UQ(self,
                     val_predictions: np.ndarray,
                     test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        (_,
         avg_last_hidden_val,
         avg_last_hidden_test) = self._compute_hidden_vals()

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        for task in range(self.args.num_tasks):
            kernel = GPy.kern.Linear(input_dim=self.args.last_hidden_size)
            gaussian = GPy.models.SparseGPRegression(
                avg_last_hidden_val,
                transformed_val[:, task:task + 1], kernel)
            gaussian.optimize()

            avg_val_preds, avg_val_var = gaussian.predict(
                avg_last_hidden_val)

            val_predictions[:, task:task + 1] = avg_val_preds
            val_uncertainty[:, task:task + 1] = np.sqrt(avg_val_var)

            avg_test_preds, avg_test_var = gaussian.predict(
                avg_last_hidden_test)

            test_predictions[:, task:task + 1] = avg_test_preds
            test_uncertainty[:, task:task + 1] = np.sqrt(avg_test_var)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_uncertainty(val_uncertainty),
                test_predictions, self._scale_uncertainty(test_uncertainty))


class MVEEstimator(UncertaintyEstimator):
    """
    An MVEEstimator alters NN structure to produce twice as many outputs.
    Half correspond to predicted labels and half correspond to uncertainties.
    """
    def __init__(self,
                 args: Namespace,
                 scaler: StandardScaler):

        super().__init__(args, scaler)

        self.sum_test_uncertainty = np.zeros(
            (len(test_data.smiles()), args.num_tasks))

    def UQ_predict(self, model: nn.Module, data_loader):

        test_preds, test_uncertainty = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            uncertainty=True
        )

        if len(test_preds) != 0:
            self.sum_test_uncertainty += np.array(test_uncertainty).clip(min=0)

    def calculate_UQ(self,
                     val_predictions: np.ndarray,
                     test_predictions: np.ndarray):

        return (val_predictions,
                np.sqrt(self.sum_val_uncertainty / self.args.ensemble_size),
                test_predictions,
                np.sqrt(self.sum_test_uncertainty / self.args.ensemble_size))


def uncertainty_estimator_builder(uncertainty_method: str):
    """
    Directory for getting the correct class of uncertainty method
    depending upon which one the user has specified.
    """

    return {
        'Dropout_VI': Dropout_VI,
        'Ensemble': Ensemble_estimator,
        'random_forest': RandomForestEstimator,
        'gaussian': GaussianProcessEstimator,
        'mve': MVEEstimator
    }[uncertainty_method]
