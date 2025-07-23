import math
from dislib.data.array import Array, array
from dislib.data.tensor import Tensor, from_pt_tensor
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, Depth, \
    COLLECTION_OUT, INOUT, COLLECTION_IN, FILE_IN
from pycompss.api.task import task
import torch
from torch import nn
import numpy as np
from dislib.pytorch import EncapsulatedFunctionsDistributedPytorch
from pycompss.api.api import compss_wait_on
from dislib.utils import train_test_split
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



@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 1},
      out_tensors={Type: COLLECTION_OUT, Depth: 1})
def _assign_blocks_to_tensors(blocks, out_tensors):
    block = np.block(blocks)
    out_tensor = torch.from_numpy(block)
    out_tensors[0] = out_tensor


@constraint(computing_units="${ComputingUnits}")
@task(tensor=COLLECTION_IN, returns=1)
def predict(neural_network, tensor):
    return neural_network.eval()(torch.FloatTensor(tensor)[0])


@constraint(computing_units="${ComputingUnits}")
@task(y_pred=COLLECTION_IN, true_labels=COLLECTION_IN, returns=1)
def squared_difference(y_pred, true_labels):
    difference = y_pred - torch.FloatTensor(true_labels)[0]
    return np.square(difference.detach().numpy())


@constraint(computing_units="${ComputingUnits}")
@task(differences=COLLECTION_IN, returns=1)
def compute_mean(differences):
    differences = np.vstack(differences)
    n_samples = differences.shape[0]
    return np.sum(differences)/n_samples


def mean_squared_error(y_predicted, true_labels):
    differences = []
    for pred, true_label in zip(y_predicted, true_labels._blocks):
        differences.append(squared_difference(pred, true_label))
    return compute_mean(differences)


def tensor_from_ds_array(ds_array, shape=None):
    if not isinstance(ds_array, Array):
        raise ValueError("The method expects to receive a ds-array.")
    new_tensor = Tensor._get_out_tensors([ds_array._n_blocks[0], 1])
    for block_i, tensor_i in zip(ds_array._blocks, new_tensor):
        _assign_blocks_to_tensors(block_i, tensor_i)
    if shape is not None and (isinstance(shape, tuple) or isinstance(shape, list)):
        return change_shape(Tensor(tensors=new_tensor,
            tensor_shape=(ds_array._reg_shape[0], ds_array.shape[1]),
            dtype=np.float64, delete=ds_array._delete), shape)
    else:
        return Tensor(tensors=new_tensor, tensor_shape=(ds_array._reg_shape[0], ds_array.shape[1]),
               dtype=np.float64, delete=ds_array._delete)

