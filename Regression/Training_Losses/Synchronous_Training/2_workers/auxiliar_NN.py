from pycompss.api.task import task
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import dislib as ds


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def assign_weights_to_model(neural_network, model_parameters):
    if hasattr(neural_network, 'neural_network_layers'):
        len_nn = len(neural_network.neural_network_layers)
        for i in range(len_nn):
            if hasattr(model_parameters.neural_network_layers[i],
                        'weight'):
                neural_network.neural_network_layers[i].weight = \
                    nn.Parameter(
                        model_parameters.neural_network_layers[i].
                        weight.float())
                neural_network.neural_network_layers[i].bias = \
                    nn.Parameter(
                        model_parameters.neural_network_layers[i].bias.
                        float())
    if hasattr(neural_network, 'dense_neural_network_layers'):
        len_nn = len(model_parameters.dense_neural_network_layers)
        for i in range(len_nn):
            if hasattr(model_parameters.dense_neural_network_layers[i], 'weight'):
                neural_network.dense_neural_network_layers[i].weight = \
                        nn.Parameter(
                                model_parameters.dense_neural_network_layers[i].weight.float())
                neural_network.dense_neural_network_layers[i].bias = \
                        nn.Parameter(
                                model_parameters.dense_neural_network_layers[i].bias.float())
    neural_network.to("cpu")
    return neural_network


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_neural_network_layers = nn.Sequential(
            nn.Linear(8, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(512, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.dense_neural_network_layers(x)
        return logits

