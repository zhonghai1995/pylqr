# pylqr
An implementation of iLQR for trajectory synthesis and control. Use finite difference to approximate gradients and hessians if they are not provided. Also support automatic differentiation with numpy from [jax](https://github.com/google/jax). Include an inverted pendulum example as the test case.

Dependencies:

Numpy

Matplotlib (Only for the test)

[jax](https://github.com/google/jax) (Only for automatic differentiation)

[pytorch](https://pytorch.org/) (Only for learning-based MPC test)
