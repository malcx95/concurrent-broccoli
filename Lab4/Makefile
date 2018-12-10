SOURCES=milli.h milli.c matrix_cpu.c

ring_cpun:
	@>&2 echo "\033[0;31mRinger cpun\033[0m"
	@gcc $(SOURCES) -o matrix_cpu
	@./matrix_cpu

ring_gpun:
	@>&2 echo "\033[0;31mRinger gpun\033[0m"
	@nvcc matrix_gpu.cu -o matrix_gpu
	@./matrix_gpu


bega_mandelbrot:
	@>&2 echo "\033[0;31mBegăr mandelbrot\033[0m"
	@g++ interactiveMandelbrot.cpp -o interactiveMandelbrot -lglut -lGL
	@./interactiveMandelbrot

gpu_bega_mandelbrot:
	@>&2 echo "\033[0;31mBegăr mandelbrot\033[0m"
	@nvcc interactiveMandelbrot.cu -o interactiveMandelbrot_gpu -lglut -lGL
	@./interactiveMandelbrot_gpu
