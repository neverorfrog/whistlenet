import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.whistlenet.config import MODEL_PARAMS as params
from core.model import Classifier

class WhistleNet(Classifier):
    def __init__(self, name, num_classes, bias=True):
        super().__init__(name, num_classes, bias=True)
        
        #Convolutional Layers (take as input the image)
        channels = params['channels']
        kernels = params['kernels']
        strides = params['strides']
        pool_kernels = params['pool_kernels']
        pool_strides = params['pool_strides']
        
        conv_layers = []
        for i in range(3):
            conv_layers.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernels[i], stride=strides[i], device=device))
            conv_layers.append(nn.BatchNorm1d(channels[i+1]))
            conv_layers.append(nn.LeakyReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=pool_kernels[i], stride=pool_strides[i]))
            conv_layers.append(nn.Dropout(params['dropout']))
        conv_layers.append(nn.Conv1d(channels[-2], channels[-1], kernel_size=kernels[-1], stride=strides[-1], device=device))
        conv_layers.append(nn.ELU())
        self.conv = nn.Sequential(*conv_layers) 

        #Fully Connected layers
        fc_dims = params['fc_dims']
        fc_layers = []
        for i in range(len(fc_dims)-1):
            fc_layers.append(nn.Linear(fc_dims[i],fc_dims[i+1],bias=bias))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(params['dropout']))
        self.fc = nn.Sequential(*fc_layers)
     
    @property               
    def loss_function(self):
        return nn.CrossEntropyLoss()

    def forward(self, x):
        '''
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        '''
        # Convolutional layers
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv(x)
        # print("state shape={0}".format(x.shape))
        
        # Fully Connected Layers
        x = torch.flatten(x,start_dim=1)
        return self.fc(x)