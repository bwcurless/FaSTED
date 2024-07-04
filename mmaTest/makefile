SOURCEDIR = .
EXE = main

SOURCES = $(wildcard $(SOURCEDIR)/*.cu)
OBJECTS = $(SOURCES:.cu=.o)

CC = nvcc

# Builds SASS for all architectures into one executable
ARCHS = -arch=compute_80 -code=sm_80

# Need to download cuda samples to here from github
LIBS="$(HOME)/cuda-samples/Common"

#
## Debug build settings
#
DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXE)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJECTS))
# rdynamic and lineinfo for running memcheck
# -keep to keep all intermediate files (like ptx)
DBGFLAGS = -std=c++17 -keep -g -G -Xptxas=-v -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -Xcompiler -rdynamic -DDEBUG

#
## Release build settings
#
RELDIR = release
RELEXE = $(RELDIR)/$(EXE)
RELOBJS = $(addprefix $(RELDIR)/, $(OBJECTS))
RELFLAGS = -std=c++17 -keep -O3 -Xptxas=-v -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

.PHONY: all clean debug prep release remake

# General rules

all: prep release

remake: clean all

prep:
	@mkdir -p $(DBGDIR) $(RELDIR)

clean:
	rm -f $(RELEXE) $(RELOBJS) $(DBGEXE) $(DBGOBJS)

# Debug rules

debug: $(DBGEXE)

$(DBGEXE): $(DBGOBJS)
	$(CC) $(DBGFLAGS) $(ARCHS) -I$(LIBS) $^ -o $@

$(DBGDIR)/%.o: %.cu
	$(CC) $(DBGFLAGS) $(ARCHS) -I$(LIBS) $< -c -o $@

# Release rules

release: $(RELEXE)

$(RELEXE): $(RELOBJS)
	$(CC) $(RELFLAGS) $(ARCHS) -I$(LIBS) $^ -o $@

$(RELDIR)/%.o: %.cu
	$(CC) $(RELFLAGS) $(ARCHS) -I$(LIBS) $< -c -o $@
