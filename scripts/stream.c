/* stream.c -- read bandwidth of this machine (sum of a 512 MB array),
 * to compare with the attraction phase.  gcc -O3 -march=native -fopenmp */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  const size_t n = 1ull << 26;                  /* 64 M doubles = 512 MB */
  double *a = malloc(n * sizeof *a);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) a[i] = 1.0;
  for (int p = 1; p <= 16; p *= 2) {
    double best = 1e9, s = 0;
    for (int r = 0; r < 5; r++) {
      double t = omp_get_wtime();
      #pragma omp parallel for schedule(static) num_threads(p) reduction(+:s)
      for (size_t i = 0; i < n; i++) s += a[i];
      t = omp_get_wtime() - t; if (t < best) best = t;
    }
    printf("p=%2d read %.1f GB/s (%g)\n", p, n * 8 / best / 1e9, s);
  }
  return 0;
}
