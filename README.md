# Continuous Convolutional Networks for Audio Detection and Classification

Project for Neural Network course at Sapienza University of Rome.
This project is about the experimentation of the contributions of 

Ingredients
- We have a one-dimensional vector-valued sequence **x**$: \mathbb{R} \rightarrow \mathbb{R}^{N_{in}}$
- And a kernel function $\psi: \mathbb{R} \rightarrow \mathbb{R}^{N_{in}}$
- $N_{in}$ is the number of channels of the input sequence
- $N_{x}$ is the length of the input sequence

Definition of convolution (with $N_{in}=1$)
- $(x * \psi)(t) = \sum\limits_{\tau=-\frac{N_x}{2}}^{\frac{N_x}{2}} x(\tau)\cdot\psi(t-\tau)$




![alt](docs/assets/discrete_conv.png)


## Continuous Kernel Convolution

The convolution operator is viewed as a vector-valued continuous function $\psi: \mathbb{R} \rightarrow \mathbb{R}^{N_{out} \times N_{in}}$, parametrized with a small neural network $MLP^{\psi}$
  - The input is a time-step of the convolvee
  - The output is the value of the convolutional kernel at that time-step

## Handling the input to $MLP^{\psi}$



## References

<a id="1">[1]</a> 
[CKConv: Continuous Kernel Convolution For Sequential Data](https://arxiv.org/pdf/2102.02611)

<a id="2">[2]</a> 
[FlexConv: Continuous Kernel Convolutions with Differentiable Kernel Sizes](https://arxiv.org/pdf/2110.08059)

<a id="3">[3]</a> 
[Modelling Long Range Dependencies in ND: From Task-Specific to a General Purpose CNN](https://arxiv.org/pdf/2301.10540)