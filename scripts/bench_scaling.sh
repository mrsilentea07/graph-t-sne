#!/usr/bin/env bash
# Run time of every phase against the number of threads p, for the given
# graphs (default: all three). Sequential (gcc and clang), OpenMP, OpenCilk,
# Pthreads.
# For k49_norm_10NN also the direct O(n^2) repulsion.
#   scripts/bench_scaling.sh k49_norm_10NN com-Amazon
cd "$(dirname "$0")/.."
OUT=results/raw/scaling.csv
GRAPHS=${*:-k49_norm_10NN com-Amazon coPapersDBLP}
PS="1 2 4 6 8 12 16"
R=7
for g in $GRAPHS; do
  extra=""; [ "$g" = k49_norm_10NN ] && extra="--direct"
  ./bin/tsne_seq       data/$g -r $R -c "$OUT" $extra -o results
  ./bin/tsne_seq_clang data/$g -r $R -c /dev/null | sed -n 2p | sed 's/^/  [clang] /'
  for p in $PS; do
    ./bin/tsne_omp data/$g -p $p -r $R -c "$OUT" $extra -o results | sed -n '1,2p;/direct O/p'
    CILK_NWORKERS=$p ./bin/tsne_cilk data/$g -r $R -c "$OUT" $extra -o results | sed -n '1,2p;/direct O/p'
    ./bin/tsne_pthreads data/$g -p $p -r $R -c "$OUT" $extra -o results | sed -n '1,2p;/direct O/p'
  done
done
