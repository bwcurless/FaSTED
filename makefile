SOURCES = utils.cu dataset.cu kernel_join.cu dmmaTensorCoresGemm.cu wmmaTensorCoresGemm.cu gpu_join.cu main.cu
OBJECTS = utils.o dataset.o kernel_join.o dmmaTensorCoresGemm.o wmmaTensorCoresGemm.o gpu_join.o main.o
CC = nvcc
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

ampere:
	echo "Compiling for Ampere generation (CC=86)"
	$(MAKE) all ARCH=compute_86 CODE=sm_86 LIBS="$(HOME)/Documents/cuda-samples/Common"

turing:
	echo "Compiling for Turing generation (CC=75)"
	$(MAKE) all ARCH=compute_75 CODE=sm_75 LIBS="$(HOME)/Documents/cuda-samples/Common"

monsoon:
	echo "Compiling for Monsoon cluster with A100 (CC=80)"
	$(MAKE) all ARCH=compute_80 CODE=sm_80 LIBS="$(HOME)/Documents/cuda-samples/Common"

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	echo $(LIBS)
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(LIBS) $^ -o $@

%.o: %.cu
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(LIBS) $^ -c $@

clean:
	rm $(OBJECTS)
	rm $(EXECUTABLE)
