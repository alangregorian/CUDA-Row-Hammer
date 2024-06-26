NVCC := nvcc
NVCCFLAGS := -O0 -Xcompiler -O0 -Xptxas -O1 -lineinfo -O0

pointer_chase: pointer_chase.cu
	$(NVCC) $(CUDAFLAGS) -o pointer_chase pointer_chase.cu

clean:
	rm -f pointer_chase