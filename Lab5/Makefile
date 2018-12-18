
ring_medianen:
	nvcc median.cu -c -g -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -g -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
	./filter

ring_gauss:
	nvcc gauss.cu -c -g -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -g -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
	./filter

ring_separabla_filtret:
	nvcc separable.cu -c -g -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -g -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
	./filter

ring_filtret:
	nvcc filter.cu -c -g -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -g -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
	./filter

ring_det_langsamma_filtret:
	nvcc original.cu -c -g -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -g -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
	./filter

