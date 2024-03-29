# AllenCahn_Equation1D
Physics informed neural network (PINN) for the 1D Allen Cahn Equation

This module implements the Physics Informed Neural Network (PINN) model for the 1D Allen Cahn equation. The Allen Cahn equation is given by (du/dt - eps d^2u/dx^2 - u^3 + u) = 0, where eps is 0.001. It has an initial condition u(t=0, x) = 0.25 sin(x). Dirichlet boundary condition is given at x = -1,+1. The PINN model predicts u(t, x) for the input (t, x).

The effectiveness of PINNs is validated in the following works.

+  M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017). (https://arxiv.org/abs/1711.10561)

+  M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017). (https://arxiv.org/abs/1711.10566)

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wave
