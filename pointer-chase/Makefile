NVCC := nvcc
NVCCFLAGS := -O0 -Xcompiler -O0 -Xptxas -O1 -lineinfo -O0

PointerChase: PointerChase.cu
	$(NVCC) $(CUDAFLAGS) -o PointerChase PointerChase.cu

clean:
	rm -f PointerChase