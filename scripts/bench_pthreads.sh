#!/usr/bin/env bash
# Pthreads runs only, appended to the existing CSVs (same settings as
# bench_scaling.sh and bench_crossover.sh), so the seq/OpenMP/OpenCilk runs
# do not have to be repeated.
cd "$(dirname "$0")/.."
R=7
for g in k49_norm_10NN com-Amazon coPapersDBLP; do
  extra=""; [ "$g" = k49_norm_10NN ] && extra="--direct"
  for p in 1 2 4 6 8 12 16; do
    ./bin/tsne_pthreads data/$g -p $p -r $R -c results/raw/scaling.csv $extra -o results | sed -n '1,2p;/direct O/p'
  done
done
./bin/tsne_pthreads data/com-Amazon -p 16 -r 3 --crossover results/raw/crossover.csv | grep crossover
