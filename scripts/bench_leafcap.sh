#!/usr/bin/env bash
# Leaf capacity m of the adaptive quadtree: time of build + com + walk and
# the error of the repulsion, sequential and OpenMP with 8 threads.
cd "$(dirname "$0")/.."
OUT=results/raw/leafcap.csv; rm -f "$OUT"
for g in k49_norm_10NN com-Amazon coPapersDBLP; do
  for m in 1 2 4 8 16 32 64 128 256; do
    ./bin/tsne_seq data/$g -m $m -r 3 -c "$OUT" > /dev/null
    ./bin/tsne_omp data/$g -m $m -p 8 -r 5 -c "$OUT" > /dev/null
  done
done
echo "wrote $OUT"
