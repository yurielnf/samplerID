# SamplerID — interpolating millions of points of low-rank functions

A C++17 library for learning tensor-network surrogates of expensive
multivariate functions `f(i_0, ..., i_{N-1}) -> T` from a small number of
evaluations.

The headline feature is a variant of **Tensor Cross Interpolation (TCI)**
allowing user-provided pivots: you can sample millions of points
separately and later combine the learned tensor trains without losing
information. The cost scales *linearly* with the number of pivots.

## What's inside

### Core decomposition
- **Interpolative decomposition (ID)** of a matrix via rank-revealing
  pivoted QR (LAPACK `geqp3`)

### Tensor trains (chain topology)
- **`TensorTrain<T>`** — basic TT container with `eval`, `sum`,
  `overlap`, and norm.
- **`tt_id<T>`** — TCI-style TT construction using row/column ID
  (instead of CUR). DMRG-style left/right sweeps.
- **`addPivotsAllBonds`** — linear-cost interpolation of a batch of
  global pivots. Sample independently, then merge.
- **`block_tt_id<T>`** — learn `n` functions simultaneously into one TT
  with a "fat" core of local dim `n * d_center` at the sweep center; all
  other cores stay the usual shape `[r, d, r']`.

### Tree tensor networks
- **`TreeTopology`** — pure topology: ordered neighbors per node, a
  root, a `phys` set tagging physical nodes. Two edge iterators 
  `leaves_to_root` and `root_to_leaves` to generate chain-like algorithms.
- **`TensorTree<T>`** — tensor-tree container with `eval`, `sum`,
  `overlap`, `norm`. All contractions are written as a chain-like sweep
  over `leaves_to_root()`: the tree recursion lives in the iterator, the
  loop body sees only one edge at a time.
- **`tree_tt_id<T>`** — ID-based construction of a tree tensor network,
  generalizing `tt_id` to arbitrary tree topologies (chain, Y-tree,
  binary tree, …). Same `addPivotsAllBonds` API for batched pivots.

### Quantics
- **Quantics grid** and **QTT** learning on top of `tt_id`.

## Building & testing

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

LibTorch is downloaded on the fly if no local `libtorch/` is found.
Test binaries: `test_interp_decomp`, `test_tt`, `test_q_tt_id`,
`test_block_tt_id`, `test_tree_tt_id`.

## Dependencies
- **LibTorch (C++)** — tensor manipulation
- **LAPACK** — pivoted QR, SVD, triangular solves
- **Catch2 v3** — testing (fetched by CMake)
