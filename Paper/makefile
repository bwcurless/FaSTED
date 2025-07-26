# Directory variables
FIGDIR=tikz_src
BUILDDIR=tikz_build

# Figures to compile (without extensions)
FIGS=Register_Based_Fragment \
Block_Tile \
Chunk_Based_Fragment \
Global_Memory_Layout \
Grid_Rasterization \
Shared_Memory_Layout \
Single_TC_MMA \
Warp_Tile

# Main document
MAIN=main

# Tools
LATEX=pdflatex
LATEXMK=latexmk

# Use standalone class for figures, output to build/
# Automatic pattern rule: build/fig1.pdf depends on figures/fig1.tex
$(BUILDDIR)/%.pdf: $(FIGDIR)/%.tex
	mkdir -p $(BUILDDIR)
	$(LATEX) -output-directory=$(BUILDDIR) $<

# Compile main.tex to build/main.pdf
$(BUILDDIR)/$(MAIN).pdf: $(MAIN).tex $(patsubst %, $(BUILDDIR)/%.pdf, $(FIGS))
	mkdir -p $(BUILDDIR)
	$(LATEXMK) -pdf -output-directory=$(BUILDDIR) $(MAIN).tex

# Default target: build everything
all: $(BUILDDIR)/$(MAIN).pdf

# Clean build files
clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean tikz
