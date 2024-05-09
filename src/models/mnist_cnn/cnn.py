import torch
import torch.nn as nn
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.mnist_cnn.config import MODEL_PARAMS as params
from core.model import Classifier

class CNN(Classifier):
    def __init__(self, name, num_classes, bias=True):
        super().__init__(name, num_classes, bias=True)
        
        self.example_args = (torch.randn(1,1,28,28),)
        self.activation = nn.LeakyReLU()
        
        #Convolutional Layers (take as input the image)
        channels = params['channels']
        kernels = params['kernels']
        strides = params['strides']
        pool_kernels = params['pool_kernels']
        pool_strides = params['pool_strides']
        
        conv_layers = []
        for i in range(len(kernels)):
            conv_layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], stride=strides[i], device=device))
            # conv_layers.append(nn.BatchNorm2d(channels[i+1]))
            conv_layers.append(self.activation)
            conv_layers.append(nn.MaxPool2d(kernel_size=pool_kernels[i], stride=pool_strides[i]))
        self.conv = nn.Sequential(*conv_layers) 

        #Fully Connected layers
        fc_dims = params['fc_dims']
        fc_layers = []
        for i in range(len(fc_dims)-1):
            fc_layers.append(nn.Linear(fc_dims[i],fc_dims[i+1],bias=bias))
            fc_layers.append(self.activation)
        fc_layers.append(nn.Dropout(params['dropout']))
        self.fc = nn.Sequential(*fc_layers)
        
        #Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for weights
                init.xavier_uniform_(m.weight)
                # Zero initialization for biases
                if m.bias is not None:
                    init.zeros_(m.bias)
     
    @property               
    def loss_function(self):
        return nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor):
        '''
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        '''
        # Convolutions
        x = x.type(torch.float).to(self.device)
        x = self.conv(x)
        # print("state shape={0}".format(x.shape))
        
        # Fully Connected Layers
        x = torch.flatten(x,start_dim=1)
        return self.fc(x)