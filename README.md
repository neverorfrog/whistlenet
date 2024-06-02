# Continuous Convolutional Neural Networks for Audio Detection and Classification

Project for Neural Networks course at Sapienza University of Rome. This project
is about the experimentation of the contributions of [[1]](#1), [[2]](#2) and [[3]](#3). The main idea is to implement a convolutional neural network architecture that uses a fully connected network to parametrize the size of the convolutional kernel itself, making the convolution in fact continuous. The main task to be solved is the detection of a whistle sound, needed for RoboCup competition. The dataset is at link [TODO mettere link].

## What is the problem with discrete convolution?

Ingredients
- We have a one-dimensional vector-valued sequence **x**$: \mathbb{R} \rightarrow \mathbb{R}^{N_{in}}$
- And a kernel function $\psi: \mathbb{R} \rightarrow \mathbb{R}^{N_{in}}$
- $N_{in}$ is the number of channels of the input sequence
- $N_{x}$ is the length of the input sequence
- $N_{k}$ is the length of the kernel

Definition of convolution (with $N_{in}=1$)
- $(x * \psi)(t) = \sum\limits_{\tau=-\frac{N_k}{2}}^{\frac{N_k}{2}} x(\tau)\cdot\psi(t-\tau)$

<p align="center">
  <img src="docs/assets/discrete_conv.png" alt="Alt text" width="300"/>
</p>

Drawbacks of this approach
- $N_k$ must be defined a priori
- No functions depending on inputs $x(t-\tau)$ for $\tau > N_k$ can be modeled (no long-range dependencies)
- Large kernels become statistically unfeasible

## Continuous Kernel Convolution

The convolution operator is viewed as a vector-valued continuous function $\psi: \mathbb{R} \rightarrow \mathbb{R}^{N_{out} \times N_{in}}$, parametrized with a small neural network $MLP^{\psi}$
  - The input is a relative position $(t-\tau)$ of the convolvee
  - The output is the value $\psi(t-\tau)$ of the convolutional kernel at that position

Important assumption: "the generated kernel is arbitrarily large"

Advantage 1: "parameterizing a convolutional kernel with a neural network is
equivalent to constructing an implicit neural representation of the kernel, with
the subtle difference that our target objective is not known a-priori, but
learned as part of the optimization task of the CNN". That means that the number
of parameters needed to represent the kernel is less than the effective
dimension of the kernel itself.

Advantage 2: The kernel can fit data of any dimensionality and resolution

Advantage 3: "CKConvs are not only more general than discrete convolutions, but
that the functional family they describe is also more general than that of
(linear) recurrent units"



### What is the sequence of relative positions we give as input to $MLP^{\psi}$

### Sine nonlinearity is the best. Why?

### Initialization?

### Choice of $\omega_0$ is important. Why?

## Continuous Convolution Block

### Residual connection

## Continuous Convolutional Neural Network


## References

<a id="1">[1]</a> 
[CKConv: Continuous Kernel Convolution For Sequential Data](https://arxiv.org/pdf/2102.02611)

<a id="2">[2]</a> 
[FlexConv: Continuous Kernel Convolutions with Differentiable Kernel Sizes](https://arxiv.org/pdf/2110.08059)

<a id="3">[3]</a> 
[Modelling Long Range Dependencies in ND: From Task-Specific to a General Purpose CNN](https://arxiv.org/pdf/2301.10540)