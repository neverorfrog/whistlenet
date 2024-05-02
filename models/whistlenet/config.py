import torch
from torch import nn

DOMAIN_PARAMS = {
    'num_classes': 2,
    'input_channels': 1
}

DATA_PARAMS = {
    'val_split_size': 0.15,
    'class_weights': [1, 30],
    'resample': True
}

TRAIN_PARAMS = {
    'max_epochs': 20,
    'learning_rate': 0.000001,
    'batch_size': 64,
    'patience': 5,
    'metrics': "f1-score",
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.0001,
    'loss_function': nn.NLLLoss()
}

MODEL_PARAMS = {
    'channels': [1,32,64,128,64],
    'kernels': [5,5,5,5],
    'strides': [2,2,2,1],
    'pool_kernels': [2,2,2,2],
    'pool_strides': [1,1,1,1],
    'fc_dims': [3520,2],
    'dropout': 0
}