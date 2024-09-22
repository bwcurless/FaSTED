all: release

release:
	nvcc main.cu -arch=compute_80 -code=sm_80 -std=c++17 -O3 -lcuda -lineinfo -I$(HOME)/cuda-samples/Common -o main
