/* experiments.c -- the extra measurements of the report (see experiments.h).
 * None of them is part of the gradient; main.c runs them only on request. */
#include "experiments.h"
#include <stdlib.h>

/* ---------------- --stats ---------------- */
/* Walks every point again (sequentially, not timed) and counts the cells
 * it visits; also counts the leaves and their points.                  */
void exp_stats(const tree_t *T, double theta) {
  int64_t tot = 0, mx = 0, leaves = 0, lp = 0;
  for (int64_t s = 0; s < T->n; s++) {
    double fx, fy, z;
    int64_t v = walk_point(T, theta * theta, s, &fx, &fy, &z);
    tot += v; if (v > mx) mx = v;
  }
  for (int64_t k = 0; k < T->nnodes; k++)
    if (T->node[k].child < 0) { leaves++; lp += T->node[k].hi - T->node[k].lo; }
  printf("  stats: cells visited per point: mean %.1f, max %lld | leaves %lld, mean points/leaf %.2f\n",
         (double)tot / T->n, (long long)mx, (long long)leaves, (double)lp / leaves);
}

/* ---------------- --direct ---------------- */
void exp_direct(const char *gname, const double *Y, int64_t n, int64_t nnz,
                const double *Frep, double Z, const double *Fref, double Zref,
                int reps, FILE *csv) {
  double *Fd = malloc(2 * n * sizeof *Fd), Zd = 0;
  double *td = malloc(reps * sizeof *td);
  for (int r = 0; r < reps; r++) {
    double t0 = wtime();
    Zd = bk_direct(Y, n, Fd);
    td[r] = wtime() - t0;
    if (csv)   /* same columns as the gradient lines; only t_walk, t_total, Z */
      fprintf(csv, "%s,%s-direct,%d,%lld,%lld,0,0,%d,0,0,0,0,0,%.6e,0,%.6e,0,0,%.15e,,,,\n",
              gname, bk_name(), bk_threads(), (long long)n, (long long)nnz, r,
              td[r], td[r], Zd);
  }
  printf("  direct O(n^2): median %.4f s | Z = %.10g | BH vs direct: Frep %.2e, Z %.2e\n",
         median(td, reps), Zd, relerr(Frep, Fd, 2 * n), fabs(Z - Zd) / Zd);
  if (Fref)
    printf("  direct vs reference (BH theta=0.1): Frep %.2e | Z %.2e\n",
           relerr(Fd, Fref, 2 * n), fabs(Zd - Zref) / Zref);
  free(Fd); free(td);
}

/* ---------------- --crossover ---------------- */
void exp_crossover(const char *gname, const double *Y, int64_t n, int leafcap,
                   double theta, int reps, const char *path) {
  FILE *fx = fopen(path, "a");
  if (fx && ftell(fx) == 0)
    fprintf(fx, "graph,version,threads,leafcap,theta,nsub,rep,t_bh,t_direct\n");
  const int64_t sizes[] = { 250, 500, 1000, 2000, 4000, 8000, 16000, 32000,
                            64000, 128000, 256000 };
  for (size_t a = 0; a < sizeof sizes / sizeof *sizes && sizes[a] <= n; a++) {
    const int64_t N = sizes[a];
    double *Ys = malloc(2 * N * sizeof *Ys), *Fs = malloc(2 * N * sizeof *Fs);
    for (int64_t k = 0; k < N; k++) {             /* evenly spaced subset */
      int64_t i = k * n / N;
      Ys[2 * k] = Y[2 * i]; Ys[2 * k + 1] = Y[2 * i + 1];
    }
    tree_t Ts; tree_alloc(&Ts, N, leafcap);
    double *tb = malloc(reps * sizeof *tb), *tdd = malloc(reps * sizeof *tdd);
    for (int r = 0; r < reps; r++) {
      times_t t = {0};
      double t0 = wtime();
      bk_tree_build(&Ts, Ys, &t); bk_tree_com(&Ts); bk_walk(&Ts, theta, Fs);
      double t1 = wtime();
      bk_direct(Ys, N, Fs);
      double t2 = wtime();
      tb[r] = t1 - t0; tdd[r] = t2 - t1;
      if (fx) fprintf(fx, "%s,%s,%d,%d,%g,%lld,%d,%.6e,%.6e\n", gname, bk_name(),
                      bk_threads(), leafcap, theta, (long long)N, r, tb[r], tdd[r]);
    }
    printf("  crossover n = %7lld: BH %.3e s, direct %.3e s\n", (long long)N,
           median(tb, reps), median(tdd, reps));
    free(tb); free(tdd); tree_free(&Ts); free(Ys); free(Fs);
  }
  if (fx) fclose(fx);
}
