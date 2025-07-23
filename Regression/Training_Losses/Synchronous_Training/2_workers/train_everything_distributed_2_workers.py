from auxiliar_NN import MLP
from auxiliar_NN_relu import assign_weights_to_model
import torch
import dislib as ds
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dislib.data import tensor_from_ds_array
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import pandas as pd
import csv
import matplotlib.pyplot as plt


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def read_dataset(dataset_file, partitions=10):
    data = pd.read_csv(dataset_file)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    x_data = data.loc[:, data.columns != 'Intensity Value 3s']
    y_data = data.loc[:, data.columns == 'Intensity Value 3s']
    x_array = ds.array(x_data, block_size=(math.ceil(x_data.shape[0]/partitions), x_data.shape[1]))
    y_array = ds.array(y_data, block_size=(math.ceil(x_data.shape[0]/partitions), 1))
    return x_array, y_array


def load_data(all_dataset, train_dataset, test_dataset):
    Data_X_arr_All, Data_Y_arr_All = read_dataset(all_dataset, partitions=20)
    x_test, y_test = read_dataset(test_dataset, partitions=4)
    x_train, y_train = read_dataset(train_dataset, partitions=4)
    minmax_scaler_total_x = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler_total_y = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler_total_x.fit(Data_X_arr_All)
    minmax_scaler_total_y.fit(Data_Y_arr_All)
    x_train = minmax_scaler_total_x.transform(x_train)
    y_train = minmax_scaler_total_y.transform(y_train)
    x_test = minmax_scaler_total_x.transform(x_test)
    y_test_scaled = minmax_scaler_total_y.transform(y_test)
    x_train = x_train.collect()
    y_train = y_train.collect()
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train.reshape(-1, 1))
    x_train = ds.from_pt_tensor(x_train, shape=(8, 1))
    y_train = ds.from_pt_tensor(y_train, shape=(8, 1))
    x_test = x_test.collect()
    y_test = y_test.collect()
    y_test_scaled = y_test_scaled.collect()
    y_test = torch.from_numpy(y_test.reshape(-1, 1))
    y_test = ds.from_pt_tensor(y_test, shape=(8, 1))
    y_test_scaled = torch.from_numpy(y_test_scaled.reshape(-1, 1))
    y_test_scaled = ds.from_pt_tensor(y_test_scaled, shape=(8, 1))
    x_test = torch.from_numpy(x_test)
    x_test = ds.from_pt_tensor(x_test, shape=(8, 1))
    return x_train, y_train, x_test, y_test, y_test_scaled, minmax_scaler_total_x, minmax_scaler_total_y


def train_main_network(x_train, y_train, x_test, y_test, scale_y_data):
    encaps_function = EncapsulatedFunctionsDistributedPytorch(num_workers=2)
    torch_model = MLP()
    torch_model.apply(init_weights)
    criterion = nn.MSELoss
    optimizer = optim.Adam
    optimizer_parameters = {"lr": 0.0002}
    encaps_function.build(torch_model, optimizer, criterion, optimizer_parameters, num_gpu=2, num_nodes=1)
    start_time = time.time()
    trained_weights, training_loss, training_r2, training_loss_deescaled, training_r2_deescaled, validation_loss, validation_loss_deescaled, \
            validation_r2, validation_r2_deescaled = encaps_function.fit_synchronous_with_GPU(x_train, y_train, 2543, 32, x_test=x_test, y_test=y_test, minmax_scaler_total_y=scale_y_data, shuffle_blocks=False, shuffle_block_data=True, return_loss=True)
    training_time = time.time() - start_time
    torch_model = assign_weights_to_model(torch_model, trained_weights)
    epochs = range(1, len(training_r2) + 1)
    plt.plot(epochs, training_r2, label='Train R2', marker='o')
    plt.plot(epochs, validation_r2, label='Validation R2', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.title('Train R2 vs Validation R2')
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.savefig('train_val_r2.png')
    plt.clf()
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, label='Train MSELoss', marker='o')
    plt.plot(epochs, validation_loss, label='Validation MSELoss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('MSELoss')
    plt.title('Train Loss vs Validation Loss')
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.savefig('train_val_loss.png')
    df = pd.DataFrame([training_loss])#[tensor.tolist() for tensor in train_loss])
    df.to_csv("train_loss.csv", index=False, decimal=",", sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([training_loss_deescaled])#[tensor.tolist() for tensor in train_loss])
    df.to_csv("train_loss_deescaled.csv", index=False, decimal=",", sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([validation_loss])
    df.to_csv("validation_loss.csv", index=False, decimal=",",sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([validation_loss_deescaled])
    df.to_csv("validation_loss_deescaled.csv", index=False, decimal=",",sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([training_r2])#tensor.tolist() for tensor in train_acc])
    df.to_csv("training_r2.csv", index=False, decimal=",", sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([training_r2_deescaled])#tensor.tolist() for tensor in train_acc])
    df.to_csv("training_r2_deescaled.csv", index=False, decimal=",", sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([validation_r2])
    df.to_csv("validation_r2.csv", index=False, decimal=",", sep=";", quoting=csv.QUOTE_NONE)
    df = pd.DataFrame([validation_r2_deescaled])
    df.to_csv("validation_r2_deescaled.csv", index=False, decimal=",", sep=";", quoting=csv.QUOTE_NONE)
    #training_time = time.time() - start_time
    return torch_model, x_train, y_train, training_time


def evaluate_main_network(x_test, y_test, torch_model):
    outputs = []
    x_test = x_test.collect()
    torch_model = torch_model.to("cuda:0")
    for x_out_tens in x_test:
        for x_in_tens in x_out_tens:
            output = torch_model(x_in_tens.float().to("cuda:0"))
            outputs.append(output)
    outputs = torch.cat(outputs)
    outputs = outputs.detach().cpu().numpy()
    outputs = ds.array(outputs, block_size=(math.ceil(outputs.shape[0]/8), outputs.shape[1]))
    outputs = minmax_scaler_total_y.inverse_transform(outputs)
    y_test = torch.cat([tens for tensor in y_test.collect() for tens in tensor])
    y_test = ds.array(y_test, block_size=(10000, 1))
    y_test = y_test.collect()
    outputs = outputs.collect()
    print("MSE: " + str(mean_squared_error(y_test, outputs)))
    print("MAE: " + str(mean_absolute_error(y_test, outputs)))
    print("R2 score: " + str(r2_score(y_test, outputs)))
    print("Pearson Corr: " + str(np.corrcoef(y_test, outputs)))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, y_test_scaled, minmax_scaler_total_x, minmax_scaler_total_y = load_data("/gpfs/scratch/bsc19/bsc019756/Intensity_Dataset/Dislib_IcelandAllData_3s.csv", 
            "/gpfs/scratch/bsc19/bsc019756/Intensity_Dataset/Dislib_Iceland_Train_RF_3s.csv", 
            "/gpfs/scratch/bsc19/bsc019756/Intensity_Dataset/Dislib_Iceland_Test_RF_3s.csv")

    model_path = "./weights/mlp_mnist.pth"
    # Original model timing
    num_epochs = 4
    # Get smaller model
    torch_model, x_train, y_train, training_time = train_main_network(x_train, y_train, x_test, y_test_scaled, minmax_scaler_total_y)

    train_data = []
    test_data = []
    print("Evaluate Original Accuracy, MSE or MAE", flush=True)
    print("Time used to train NN: " + str(training_time))
    evaluate_main_network(x_test, y_test, torch_model)

