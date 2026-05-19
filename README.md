# Interpolating millions of points for a low-rank function 
- We implement a variant of Tensor Cross Interpolation designed for learning a huge number of user-provided pivots.
- We can sample millions of points separately, and later combine the learned tensor trains without losing information. 
- The cost scales linear with the number of pivots.

# Features
We provide: 
- A matrix interpolative decomposition (ID) based on rank revealing QR (from lapack)
- A basic `TensorTrain` class
- A class doing something like Tensor Cross Interpolation (TCI) , but using ID instead of CUR
- A new implementation of linear-cost interpolation of global pivots
- The quantics grid and qtt learning

# Dependencies
- `libtorch c++` for tensor manipulation
- `Catch2` for testing
- `lapack` for linear algebra
