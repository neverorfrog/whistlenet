# Notes

- Why?
  - Obtain a formulation for a CNN applicable to arbitrary resolutions and sizes
  - Model the long-range dependencies needed to extract high-level features in a parameter-efficient way
- How?
  - Use the model weights to parameterize k as a continuous function over the data domain $\R^d$
  - We require a parameterization for convolutional layers that is invariant to the set $\Omega(x)$ over which input $x$ is sampled
  - To avoid models with different parameter count for different resolutions, it is necessary that the parameterization of the kernel decouples its parameter count from the size of the kernel.
  - With this parametrization we model the kernel over the underlying continuous domain $\R^d$ of the input signal
  - This also removes the need for downsampling and stacking of layers to increase receptive fields

## FlexConvs

- FlexConvs deﬁne their convolutional kernels ψ as the product of the output of a neural network MLPψ with a Gaussian mask of local support. The neural network MLPψ parameterizes the kernel, and the Gaussian mask parameterizes its size

## CKConvs

- Provide a continuous parameterization for convolutional kernels by using a small neural network as a kernel generator network
