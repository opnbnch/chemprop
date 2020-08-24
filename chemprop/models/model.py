import torch.nn as nn

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function, initialize_weights
from torch import var, mean, tensor, index_select, stack


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(MoleculeModel, self).__init__()

        self.args = args
        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer
        self.uncertainty = args.uncertainty
        self.mve = args.uncertainty == 'mve'
        self.use_last_hidden = True
        self.two_outputs = args.uncertainty == 'Dropout_VI' or args.uncertainty == 'Ensemble'

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if self.mve:
            self.output_size *= 2

        args.last_hidden_size = self.output_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,

            ]
            last_linear_dim = first_linear_dim
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
                ])
            ffn.extend([
                activation,
                dropout,

            ])
            last_linear_dim = args.ffn_hidden_size

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        if self.two_outputs:
            self.logvar_layer = nn.Linear(last_linear_dim, self.output_size)

        self.output_layer = nn.Linear(last_linear_dim, self.output_size)

    def featurize(self, *input):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(*input))

    def get_estimates(self, fork):
        """
        Computes the variance and mean of the final layer before activation
        :param input: Input
        """

        return var(fork, dim=1), mean(fork, dim=1)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """

        _output = self.ffn(self.encoder(*input))

        if self.two_outputs:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)

            return output, logvar

        if self.mve:
            _output = self.output_layer(_output)
            even_indices = tensor(range(0, list(_output.size())[1], 2))
            odd_indices = tensor(range(1, list(_output.size())[1], 2))

            if self.args.cuda:
                even_indices = even_indices.cuda()
                odd_indices = odd_indices.cuda()

            predicted_means = index_select(_output, 1, even_indices)
            predicted_uncertainties = index_select(_output, 1, odd_indices)
            capped_uncertainties = nn.functional.softplus(predicted_uncertainties)

            output = stack((predicted_means, capped_uncertainties), dim=2).view(_output.size())
            return output

        if self.featurizer:
            return self.featurize(*input)

        # BUG: Turns off if we use multiple folds or ensemble
        if self.use_last_hidden:
            output = self.output_layer(_output)
        else:
            return _output

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output
