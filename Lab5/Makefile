
ring_filtret:
	nvcc filter.cu -c -g -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -g -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
	./filter

