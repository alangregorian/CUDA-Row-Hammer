NVCC := nvcc
NVCCFLAGS := -O0 -Xcompiler -O0 -Xcompiler -Wextra -Xcompiler -Wall -lcuda

TURINGAS := python3 -m turingas.main

.PHONY: all clean

all: strided_access strided_access.cubin

strided_access: strided_access.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 $< -o $@

strided_access.cubin: strided_access.sass
	$(TURINGAS) -arch 80 -i $< -name strided_access -o $@

clean:
	rm -f strided_access strided_access.cubin