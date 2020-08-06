import torch.nn as nn

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function, initialize_weights, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout, RegularizationAccumulator
from torch import var, mean


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

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer
        self.uncertainty = args.uncertainty

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)
        self.dropout_vi = args.uncertainty == 'Dropout_VI'

        if self.dropout_vi:
            args.reg_acc = RegularizationAccumulator()
            args.reg_acc.initialize(cuda=args.cuda)

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

        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                ConcreteDropout(layer=nn.Linear(first_linear_dim, args.ffn_hidden_size),
                                reg_acc=args.reg_acc, weight_regularizer=wd,
                                dropout_regularizer=dd) if self.dropout_vi else
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
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
                    ConcreteDropout(layer=nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                                    reg_acc=args.reg_acc, weight_regularizer=wd,
                                    dropout_regularizer=dd) if self.mc_dropout else
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
                ])
            ffn.extend([
                activation,
                dropout,

            ])
            last_linear_dim = args.ffn_hidden_size

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        if self.uncertainty:
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

        if self.featurizer:
            return self.featurize(*input)

        if self.uncertainty:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)

            return output, logvar
        else:
            output = self.output_layer(_output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
 
        return output
