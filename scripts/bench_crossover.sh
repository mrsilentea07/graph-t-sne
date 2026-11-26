#!/usr/bin/env bash
# Barnes-Hut (build + centers + walk) vs direct O(n^2) on subsets of n points.
cd "$(dirname "$0")/.."
OUT=results/raw/crossover.csv; rm -f "$OUT"
./bin/tsne_seq data/com-Amazon -r 3 --crossover "$OUT" | grep crossover
./bin/tsne_omp data/com-Amazon -p 16 -r 3 --crossover "$OUT" | grep crossover
CILK_NWORKERS=16 ./bin/tsne_cilk data/com-Amazon -r 3 --crossover "$OUT" | grep crossover
./bin/tsne_pthreads data/com-Amazon -p 16 -r 3 --crossover "$OUT" | grep crossover
