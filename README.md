# StateSpace-python
A state space modeling class in python with a single time step solver to ease integration with controllers. The solution technique was discussed in my paper [Linear Time Invariant State Space System Identification Using Adam Optimization](https://ieeexplore.ieee.org/document/9047808/).

You will find Examples.ipynb in [the examples repository](https://github.com/MarkNaeem/StateSpace-python-examples) where some examples shows how to use the class.

### Why use it?

The advantage of this class is that it can take input one step at a time. So if you are developing a controller that takes  feedback form this model to adjust its output in the next iteration, this model can do this (compared to scipy model which solves for the input points all at once).
