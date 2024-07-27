NVCC := nvcc
NVCCFLAGS := -O0 -Xcompiler -O0 -Xcompiler -Wextra -Xcompiler -Wall -Xptxas -O1

.PHONY: all clean

all: pointer_chase

pointer_chase: pointer_chase.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f pointer_chase
