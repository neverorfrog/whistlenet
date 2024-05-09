from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from src.tests.function_fitting.functions import example_function

# Load the configuration
file_path = "function_fitting/config.yaml"
with open(file_path, 'r') as file:
    config = OmegaConf.create(yaml.safe_load(file))

# Set the seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Get the function
x = ( torch.linspace(config.min, config.max, config.samples)
      .type(torch.FloatTensor).unsqueeze(0).unsqueeze(0) )

function = example_function(config)
f = ( torch.from_numpy(function)
      .type(torch.FloatTensor).unsqueeze(0).unsqueeze(0) )
plt.plot(function)
plt.title("Ground Truth")
plt.show()

# Get the model

