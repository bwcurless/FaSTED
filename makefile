SOURCEDIR = .
BUILDDIR = build
EXECUTABLE = main

SOURCES = $(wildcard $(SOURCEDIR)/*.cu)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cu,./$(BUILDDIR)/%.o,$(SOURCES))

CC = nvcc

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
# rdynamic and lineinfo for running memcheck
DEBUGFLAGS = -Xcompiler -rdynamic -lineinfo

ARCHS = -arch=sm_75 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \

# Need to download cuda samples to here from github
LIBS="$(HOME)/Documents/cuda-samples/Common"


all: prep $(BUILDDIR)/$(EXECUTABLE)

.PHONY: prep
prep:
	@mkdir -p $(BUILDDIR)

$(BUILDDIR)/$(EXECUTABLE): $(OBJECTS)
	echo $(LIBS)
	$(CC) $(FLAGS) $(ARCHS) -I$(LIBS) $^ -o $@

$(OBJECTS): $(BUILDDIR)/%.o: %.cu
	$(CC) $(FLAGS) $(ARCHS) -I$(LIBS) $< -c -o $@

clean:
	rm -rf $(BUILDDIR)
