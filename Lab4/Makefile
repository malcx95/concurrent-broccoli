SOURCES=milli.h milli.c matrix_cpu.c

ring_cpun:
	@>&2 echo "\033[0;31mRinger cpun\033[0m"
	@gcc $(SOURCES) -o matrix_cpu
	@./matrix_cpu

ring_gpun:
	@>&2 echo "\033[0;31mRinger gpun\033[0m"
	@nvcc matrix_gpu.cu -o matrix_gpu
	@./matrix_gpu

