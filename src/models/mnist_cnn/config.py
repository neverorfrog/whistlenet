import torch
from torchvision import transforms
from torch import nn

DOMAIN_PARAMS = {
    'num_classes': 10,
    'input_channels': 1
}

DATA_PARAMS = {
    'data_transform': transforms.Compose([transforms.ToTensor()]),
    'val_split_size': 0.15
}

TRAIN_PARAMS = {
    'max_epochs': 70,
    'learning_rate': 0.001,
    'batch_size': 64,
    'patience': 5,
    'metrics': "f1-score",
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.001,
    'loss_function': nn.NLLLoss()
}

MODEL_PARAMS = {
    'channels': [DOMAIN_PARAMS['input_channels'],10,20],
    'kernels': [5,5],
    'strides': [1,1],
    'pool_kernels': [2,2],
    'pool_strides': [2,2],
    'fc_dims': [320,DOMAIN_PARAMS['num_classes']],
    'dropout': 0.1
}