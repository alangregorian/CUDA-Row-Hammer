# CUDA Row Hammer
This repository contains a collection of tests that attempt to bypass the L2 cache of Ada architecture GPUs and benchmark GDDR6 memory. These tests were created to verify the row size and determine the open/closed row policy -- information that would be used to identify Row Hammer vulnerable memory access patterns.

## Requirements
* CUDA 11.0+

## Project Structure
* `dump/` -
* `last_attempt_timing/` -
* `pointer_chase/` - CUDA pointer chasing test
* `strided_access/` - SaSS strided access test
* `tests/` -

## References
1. X. Mei and X. Chu, ‘Dissecting GPU Memory Hierarchy Through Microbenchmarking’, IEEE Transactions on Parallel and Distributed Systems, vol. 28, no. 1, pp. 72–86, 2017.
2. D. Yan, W. Wang, and X. Chu, ‘Optimizing batched winograd convolution on GPUs’, in Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, San Diego, California, 2020, pp. 32–44.
