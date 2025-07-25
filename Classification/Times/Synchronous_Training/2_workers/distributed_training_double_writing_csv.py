from NFNet import NFNet, init_weights, assign_weights_to_model
import torch
import dislib as ds
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dislib.preprocessing import MinMaxScaler
import math
from dislib.pytorch import EncapsulatedFunctionsDistributedPytorch
from dislib.data.array import Array
from dislib.data.tensor import Tensor

from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, Depth, \
    INOUT, IN, COLLECTION_OUT, COLLECTION_IN
from pycompss.api.task import task
from dislib.data.tensor import load_dataset

from pycompss.api.api import compss_wait_on
import copy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv


def process_outputs(output_nn):
    _, indices = torch.max(output_nn, dim=1)
    binary_output = torch.zeros_like(output_nn)
    binary_output[torch.arange(output_nn.size(0)), indices] = 1
    return binary_output

def load_data(x_train, y_train, x_test, y_test):
    x_train = torch.load(x_train).float()
    x_train = ds.from_pt_tensor(x_train, shape=(8, 1))
    y_train = torch.load(y_train).float()
    y_train = ds.from_pt_tensor(y_train, shape=(8, 1))
    x_test = torch.load(x_test)
    x_test = ds.from_pt_tensor(x_test, shape=(8, 1))
    y_test = torch.load(y_test)
    y_test = ds.from_pt_tensor(y_test, shape=(8, 1))
    return x_train, y_train, x_test, y_test


def train_main_network(x_train, y_train, x_test, y_test):
    encaps_function = EncapsulatedFunctionsDistributedPytorch(num_workers=2)
    torch_model = NFNet().to("cuda:0")
    torch_model.apply(init_weights)
    criterion = nn.CrossEntropyLoss
    optimizer = optim.SGD
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer_parameters = {"lr": 0.1, "momentum": 0.9, "weight_decay":0.0001}
    indexes=128
    num_batches = math.ceil(x_train.tensor_shape[0]/indexes)
    encaps_function.build(torch_model, optimizer, criterion, optimizer_parameters, scheduler=scheduler, T_max=100, eta_min=0, num_gpu=2, num_nodes=1) 
    start_time = time.time()
    trained_weights = encaps_function.fit_synchronous_with_GPU(x_train, y_train, num_batches, 50) 
    training_time = time.time() - start_time
    torch_model = assign_weights_to_model(torch_model, trained_weights)
    return torch_model, training_time


def evaluate_main_network(x_test, y_test, torch_model):
    outputs = []
    x_test = x_test.collect()
    torch.cuda.empty_cache()
    torch_model = torch_model.eval().to("cuda:0")
    indexes = 128
    num_batches = math.ceil(x_test[0][0].shape[0]/indexes)
    for x_out_tens in x_test:
        for x_in_tens in x_out_tens:
            x_in_tens = x_in_tens.to("cuda:0")
            for idx in range(num_batches):
                with torch.no_grad():
                    output = torch_model(x_in_tens[idx * indexes: (idx + 1) *indexes].float())
                output_cpu = output.cpu()
                outputs.append(output_cpu)
                del output
            x_in_tens = x_in_tens.to("cpu")
            del x_in_tens
            torch.cuda.empty_cache()
    outputs = torch.cat(outputs)
    y_test = torch.cat([tens for tensor in y_test.collect() for tens in tensor])
    y_test = ds.array(y_test, block_size=(15000, 10))
    y_test = y_test.collect()
    outputs = process_outputs(outputs)
    outputs = outputs.detach().cpu().numpy()
    print("Accuracy: " + str(accuracy_score(y_test, outputs)))
    print("Recall: " + str(recall_score(y_test, outputs, average=None)))
    print("Precision: " + str(precision_score(y_test, outputs, average=None)))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data("/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/Dataset/train_valid/x_vt_64.pt", 
            "/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/Dataset/train_valid/y_vt.pt", 
            "/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/Dataset/test/x_test_64.pt", "/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/Dataset/test/y_test_one_hot_encoded.pt")

    model_path = "./weights/mlp_mnist.pth"
    # Original model timing
    num_epochs = 4
    # Get smaller model
    print("IS IT USING PYTORCH CACHE?")
    print(torch.backends.cudnn.enabled)
    torch_model, training_time = train_main_network(x_train, y_train, x_test, y_test)
    print("IS IT USING PYTORCH CACHE?")
    print(torch.backends.cudnn.enabled)

    train_data = []
    test_data = []
    print("Evaluate Original Accuracy, MSE or MAE", flush=True)
    print("Time used to train NN: " + str(training_time))
    evaluate_main_network(x_test, y_test, torch_model)

