# Homework 1 — how to build and run

```bash
make                       # bin/tsne_seq, bin/tsne_seq_clang, bin/tsne_omp, bin/tsne_cilk, bin/tsne_pthreads, bin/compare
./bin/tsne_seq  data/k49_norm_10NN --direct --stats
./bin/tsne_omp  data/coPapersDBLP -p 8 -r 5 -o results
CILK_NWORKERS=8 ./bin/tsne_cilk data/coPapersDBLP -r 5 -o results
./bin/tsne_pthreads data/coPapersDBLP -p 8 -r 5 -o results
./bin/compare results/coPapersDBLP-seq.mat results/coPapersDBLP-omp.mat
OMP_SCHEDULE=static ./bin/tsne_omp data/coPapersDBLP -p 8   # change the row schedule
```
Options: `-p` threads (OpenMP, Pthreads), `-r` repetitions, `-m` leaf capacity (default 16),
`-o DIR` save results, `-c FILE` append timings (CSV), `--direct`,
`--stats` (cells visited per point), `--crossover FILE`.

| path | content |
|---|---|
| `src/tsne.h` | data structures + shared per-row / per-point kernels |
| `src/main.c` | driver: read, time each phase, check vs reference, save |
| `src/experiments.c` | the extra measurements: `--stats`, `--direct`, `--crossover` |
| `src/tsne_seq.c`, `src/tsne_omp.c`, `src/tsne_cilk.c`, `src/tsne_pthreads.c` | the four versions |
| `src/compare.c` | checks that two result files agree |
| `scripts/bench_*.sh` | experiments: `leafcap`, `scaling`, `sched`, `crossover`; `pthreads` = only the Pthreads runs, appended |
| `scripts/overhead.c` | cost of a job / barrier: Pthreads pool vs OpenMP (`results/raw/overhead.txt`) |
| `scripts/predict.py`, `scripts/plots.py`, `scripts/stream.c` | prediction, figures, bandwidth test |
| `results/<graph>-<version>.mat` | beta, Fattr, Frep, Z, grad for each graph and version |
| `results/raw/*.csv` | raw timings of every run (data behind every table and plot) |
| `report/report.tex`, `report/report.pdf` | report (all four versions) |

Reproduce everything: `scripts/bench_leafcap.sh; scripts/bench_scaling.sh k49_norm_10NN com-Amazon;
python3 scripts/predict.py; scripts/bench_scaling.sh coPapersDBLP; scripts/bench_sched.sh;
scripts/bench_crossover.sh; python3 scripts/plots.py; (cd report && pdflatex report.tex)`
