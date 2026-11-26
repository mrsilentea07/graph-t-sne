/* compare.c -- check that two result files agree.
 *   ./bin/compare results/k49_norm_10NN-seq.mat results/k49_norm_10NN-omp.mat
 * Prints max |a - b| / max |a| for beta, Fattr, Frep, grad, and the
 * relative difference of Z. Exit status 1 if any exceeds 1e-10.        */
#include <matio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double *rd(mat_t *f, const char *name, size_t *len) {
  matvar_t *v = Mat_VarRead(f, name);
  if (!v) { fprintf(stderr, "missing %s\n", name); exit(2); }
  size_t n = 1;
  for (int d = 0; d < v->rank; d++) n *= v->dims[d];
  double *x = malloc(n * sizeof *x);
  for (size_t k = 0; k < n; k++) x[k] = ((double *)v->data)[k];
  Mat_VarFree(v);
  *len = n;
  return x;
}

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s a.mat b.mat\n", argv[0]); return 2; }
  mat_t *a = Mat_Open(argv[1], MAT_ACC_RDONLY), *b = Mat_Open(argv[2], MAT_ACC_RDONLY);
  if (!a || !b) { fprintf(stderr, "cannot open inputs\n"); return 2; }
  const char *names[] = { "beta", "Fattr", "Frep", "grad", "Z" };
  int bad = 0;
  printf("%s vs %s:", argv[1], argv[2]);
  for (int v = 0; v < 5; v++) {
    size_t na, nb;
    double *x = rd(a, names[v], &na), *y = rd(b, names[v], &nb);
    if (na != nb) { printf(" %s: sizes differ\n", names[v]); return 1; }
    double e = 0, s = 0;
    for (size_t k = 0; k < na; k++) {
      if (fabs(x[k] - y[k]) > e) e = fabs(x[k] - y[k]);
      if (fabs(x[k]) > s) s = fabs(x[k]);
    }
    double r = s > 0 ? e / s : e;
    printf(" %s %.1e", names[v], r);
    if (r > 1e-10) bad = 1;
    free(x); free(y);
  }
  printf("  -> %s\n", bad ? "DIFFER" : "agree");
  Mat_Close(a); Mat_Close(b);
  return bad;
}
