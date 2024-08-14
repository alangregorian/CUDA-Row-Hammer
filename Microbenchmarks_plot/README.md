# Microbenchmarks Plot
This script was used to plot the cache and memory latency of the Ada architecture GPU we targeted, which helped us verify the L1 and L2 cache sizes. It uses the same set of benchmarks [1] from the article “Microbenchmarking Nvidia’s RTX 4090” [2].

## Building & Running Microbenchmarks
```
# Clone the Microbenchmarks source code
git clone https://github.com/clamchowder/Microbenchmarks.git
cd Microbenchmarks/GpuMemLatency

# Compile the latency benchmark
cp kernels/unrolled_latency_test.cl .
make all

# Run the benchmark and store the results
./GpuMemLatency_amd64 > ../../GpuMemLatency_amd64.out

# Return to the Python script directory
cd ../../
```

## Running Python Script
```
python3 plot_latency.py GPUMemLatency_amd64.out
```

## References
1. https://github.com/clamchowder/Microbenchmarks
2. https://chipsandcheese.com/2022/11/02/microbenchmarking-nvidias-rtx-4090/
