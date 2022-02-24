# REN
Recurrent Equilibrium Networks.
This project contains the code for running the experiments from the following papers:  
-[Recurrent Equilibrium Networks: Unconstrained Learning of Stable and Robust Dynamical Models](http://128.84.4.34/abs/2104.05942v1)  
-[Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness](http://128.84.4.34/abs/2104.05942)  
-[Learning over All Stabilizing Nonlinear Controllers for a Partially-Observed Linear System](https://arxiv.org/abs/2112.04219)  
 
Recurrent Equilbrium Networks are a parametrization of recurrent neural network that obey prescribed integral quadratic constraints. 
This allows for properties such as stability or gain bounds to be imposed on the resulting networks. Such properties are useful when learning dynamical systems or controllers.

All the code is written in Julia. You should be able to install all necessary packages by instantiating. I.e., in the Julia REPL, type: "] instantiate" and all packages should be installed.
