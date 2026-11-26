# Makefile -- PDS Homework 1, graph t-SNE with Barnes-Hut.
#
#   make            all versions
#   make seq        sequential (gcc)          -> bin/tsne_seq
#   make seq_clang  sequential (OpenCilk clang, to separate compiler effects)
#   make omp        OpenMP (gcc -fopenmp)     -> bin/tsne_omp
#   make cilk       OpenCilk (clang -fopencilk) -> bin/tsne_cilk
#   make pthreads   Pthreads (gcc -pthread)   -> bin/tsne_pthreads
#   make compare    tool to compare two result files -> bin/compare
#   make clean

CC       ?= gcc
CILKCC   ?= /opt/opencilk-3.0.0/bin/clang
OPT      := -O3 -march=native
WARN     := -Wall -Wextra -Wno-unused-parameter
MATIO_CFLAGS := $(shell pkg-config --cflags matio 2>/dev/null)
MATIO_LIBS   := $(shell pkg-config --libs matio 2>/dev/null || echo -lmatio)

COMMON_SRC := src/main.c src/experiments.c starter/data.c
HDR        := src/tsne.h src/experiments.h starter/data.h

all: seq seq_clang omp cilk pthreads compare

bin:
	mkdir -p bin

seq: bin/tsne_seq
omp: bin/tsne_omp
cilk: bin/tsne_cilk
pthreads: bin/tsne_pthreads
seq_clang: bin/tsne_seq_clang
compare: bin/compare

bin/tsne_seq: $(COMMON_SRC) src/tsne_seq.c $(HDR) | bin
	$(CC) $(OPT) $(WARN) $(MATIO_CFLAGS) -o $@ $(COMMON_SRC) src/tsne_seq.c $(MATIO_LIBS) -lm

bin/tsne_seq_clang: $(COMMON_SRC) src/tsne_seq.c $(HDR) | bin
	$(CILKCC) $(OPT) $(WARN) $(MATIO_CFLAGS) -o $@ $(COMMON_SRC) src/tsne_seq.c $(MATIO_LIBS) -lm

bin/tsne_omp: $(COMMON_SRC) src/tsne_omp.c $(HDR) | bin
	$(CC) $(OPT) $(WARN) -fopenmp $(MATIO_CFLAGS) -o $@ $(COMMON_SRC) src/tsne_omp.c $(MATIO_LIBS) -lm

bin/tsne_cilk: $(COMMON_SRC) src/tsne_cilk.c $(HDR) | bin
	$(CILKCC) $(OPT) $(WARN) -fopencilk $(MATIO_CFLAGS) -o $@ $(COMMON_SRC) src/tsne_cilk.c $(MATIO_LIBS) -lm

bin/tsne_pthreads: $(COMMON_SRC) src/tsne_pthreads.c $(HDR) | bin
	$(CC) $(OPT) $(WARN) -pthread $(MATIO_CFLAGS) -o $@ $(COMMON_SRC) src/tsne_pthreads.c $(MATIO_LIBS) -lm

bin/compare: src/compare.c | bin
	$(CC) -O2 $(WARN) $(MATIO_CFLAGS) -o $@ src/compare.c $(MATIO_LIBS) -lm

clean:
	rm -rf bin

.PHONY: all seq omp cilk pthreads seq_clang compare clean
