SOURCEDIR = .
BUILDDIR = build
EXECUTABLE = main

#SOURCES = utils.cu dataset.cu kernel_join.cu dmmaTensorCoresGemm.cu wmmaTensorCoresGemm.cu gpu_join.cu main.cu
#OBJECTS = utils.o d/ataset.o kernel_join.o dmmaTensorCoresGemm.o wmmaTensorCoresGemm.o gpu_join.o main.o
SOURCES = $(wildcard $(SOURCEDIR)/*.cu)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cu,./$(BUILDDIR)/%.o,$(SOURCES))

CC = nvcc

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
# rdynamic and lineinfo for running memcheck
DEBUGFLAGS = -std=c++14 -Xcompiler -fopenmp -lcuda -Xcompiler -rdynamic -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

ampere:
	echo "Compiling for Ampere generation (CC=86)"
	$(MAKE) all ARCH=compute_86 CODE=sm_86 LIBS="$(HOME)/Documents/cuda-samples/Common"

turing:
	echo "Compiling for Turing generation (CC=75)"
	$(MAKE) all ARCH=compute_75 CODE=sm_75 LIBS="$(HOME)/Documents/cuda-samples/Common"

monsoon:
	echo "Compiling for Monsoon cluster with A100 (CC=80)"
	$(MAKE) all ARCH=compute_80 CODE=sm_80 LIBS="$(HOME)/Documents/cuda-samples/Common"

all: build $(BUILDDIR)/$(EXECUTABLE)

build:
	mkdir $(BUILDDIR)

$(BUILDDIR)/$(EXECUTABLE): $(OBJECTS)
	echo $(LIBS)
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(LIBS) $^ -o $@

$(OBJECTS): $(BUILDDIR)/%.o: %.cu
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(LIBS) $< -c --output-file $@

clean:
	rm -rf $(BUILDDIR)
