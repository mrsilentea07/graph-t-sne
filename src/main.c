/* main.c -- driver shared by all versions.
 *
 *   ./bin/tsne_<version> data/<graph> [options]
 *     -p P        threads (OpenMP, Pthreads; for OpenCilk set CILK_NWORKERS)
 *     -r R        repetitions of the whole gradient (default 5)
 *     -m M        leaf capacity of the quadtree (default 16)
 *     -o DIR      save beta, Fattr, Frep, Z, grad in DIR/<graph>-<version>.mat
 *     -c FILE     append one line of timings per repetition to FILE (CSV)
 *     --direct    also compute the repulsion directly, O(n^2)
 *     --stats     print quadtree statistics (cells visited per point)
 *     --crossover FILE  time Barnes-Hut vs direct on subsets of n, CSV to FILE
 *
 * Steps: read the data (not timed) -> R times the gradient, every phase
 * timed separately -> print the median times -> compare with the
 * reference -> optional experiments (experiments.c) -> save.
 */
#include "tsne.h"
#include "experiments.h"
#include "../starter/data.h"
#include <matio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================== */
/*                               helpers                               */
/* ================================================================== */
double wtime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int cmp_dbl(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
double median(double *x, int r) {
  qsort(x, r, sizeof *x, cmp_dbl);
  return r % 2 ? x[r / 2] : 0.5 * (x[r / 2 - 1] + x[r / 2]);
}

/* max |x - xref| */
static double maxdiff(const double *x, const double *xr, int64_t m) {
  double e = 0;
  for (int64_t k = 0; k < m; k++) if (fabs(x[k] - xr[k]) > e) e = fabs(x[k] - xr[k]);
  return e;
}
/* max |x - xref| / max |xref| */
double relerr(const double *x, const double *xr, int64_t m) {
  double s = 0;
  for (int64_t k = 0; k < m; k++) if (fabs(xr[k]) > s) s = fabs(xr[k]);
  double e = maxdiff(x, xr, m);
  return s > 0 ? e / s : e;
}

static const char *basename_of(const char *p) {     /* data/k49 -> k49 */
  const char *s = strrchr(p, '/');
  return s ? s + 1 : p;
}

/* ---------------- tree memory ---------------- */
void tree_alloc(tree_t *T, int64_t n, int leafcap) {
  memset(T, 0, sizeof *T);
  T->n = n; T->leafcap = leafcap;
  T->key  = malloc(n * sizeof *T->key);
  T->perm = malloc(n * sizeof *T->perm);
  T->pts  = malloc(2 * n * sizeof *T->pts);
  T->cap  = n / (leafcap > 1 ? leafcap : 1) * 2 + 64;
  T->node = malloc(T->cap * sizeof *T->node);
}
void tree_reserve(tree_t *T, int64_t need) {        /* grow by doubling */
  if (need <= T->cap) return;
  while (T->cap < need) T->cap *= 2;
  T->node = realloc(T->node, T->cap * sizeof *T->node);
  if (!T->node) { fprintf(stderr, "out of memory (tree)\n"); exit(1); }
}
void tree_free(tree_t *T) {
  free(T->key); free(T->perm); free(T->pts); free(T->node);
}

/* ---------------- graph (CSC from the file) -> CSR ---------------- */
/* W is symmetric, so column j of the file is row j: same arrays, with
 * 32-bit column indices (half the memory traffic of 64-bit ones).      */
static void to_csr(const graph_t *G, csr_t *W) {
  W->n = G->n; W->nnz = G->nnz;
  W->rowptr = malloc((G->n + 1) * sizeof *W->rowptr);
  W->col = malloc(G->nnz * sizeof *W->col);
  W->w = malloc(G->nnz * sizeof *W->w);
  memcpy(W->rowptr, G->colptr, (G->n + 1) * sizeof(int64_t));
  for (int64_t k = 0; k < G->nnz; k++) { W->col[k] = (int32_t)G->row[k]; W->w[k] = G->val[k]; }
  /* rows must be sorted for the binary search of the transpose entry */
  for (int64_t i = 0; i < W->n; i++)
    for (int64_t k = W->rowptr[i] + 1; k < W->rowptr[i + 1]; k++)
      if (W->col[k] <= W->col[k - 1]) {
        fprintf(stderr, "error: row %lld is not sorted\n", (long long)i);
        exit(1);
      }
}

/* ---------------- save the results (.mat, v7.3) ---------------- */
static void put(mat_t *f, const char *name, double *x, size_t rows, size_t cols) {
  size_t dims[2] = { rows, cols };
  matvar_t *v = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, x, 0);
  Mat_VarWrite(f, v, MAT_COMPRESSION_ZLIB);
  Mat_VarFree(v);
}
static void save(const char *path, int64_t n, double *beta, double *Fattr,
                 double *Frep, double Z, double *grad) {
  mat_t *f = Mat_CreateVer(path, NULL, MAT_FT_MAT73);
  if (!f) { fprintf(stderr, "cannot create %s\n", path); return; }
  put(f, "beta", beta, n, 1);
  put(f, "Fattr", Fattr, 2, n);
  put(f, "Frep", Frep, 2, n);
  put(f, "grad", grad, 2, n);
  put(f, "Z", &Z, 1, 1);
  Mat_Close(f);
}

/* ================================================================== */
/*                                main                                 */
/* ================================================================== */
int main(int argc, char **argv) {
  /* ---- 1. options ---- */
  if (argc < 2) {
    fprintf(stderr, "usage: %s data/<graph> [-p P] [-r R] [-m M] [-o DIR] [-c FILE] "
                    "[--direct] [--stats] [--crossover FILE]\n", argv[0]);
    return 1;
  }
  const char *base = argv[1]; //πχ --> ποιον γραφο θα διαβασει
  int p = 0, reps = 5, leafcap = 16, do_direct = 0, do_stats = 0;
  const char *outdir = NULL, *csv = NULL, *xover = NULL;
  for (int a = 2; a < argc; a++) {
    if (!strcmp(argv[a], "-p") && a + 1 < argc) p = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-r") && a + 1 < argc) reps = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-m") && a + 1 < argc) leafcap = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-o") && a + 1 < argc) outdir = argv[++a];
    else if (!strcmp(argv[a], "-c") && a + 1 < argc) csv = argv[++a];
    else if (!strcmp(argv[a], "--direct")) do_direct = 1;
    else if (!strcmp(argv[a], "--stats")) do_stats = 1;
    else if (!strcmp(argv[a], "--crossover") && a + 1 < argc) xover = argv[++a];
    else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
  }
  if (reps < 1) reps = 1;
  if (leafcap < 1) leafcap = 1;

  /* ---- 2. read the data (not timed) ---- */
  graph_t G; state_t S; ref_t R;
  if (read_graph(base, &G) || read_state(base, &S)) return 1;
  const int have_ref = (read_ref(base, &R) == 0);
  if (S.n != G.n) { fprintf(stderr, "sizes do not match\n"); return 1; }
  csr_t W; to_csr(&G, &W);
  free_graph(&G);
  const int64_t n = W.n;
  const double *Y = S.Y0;
  const double theta = S.theta;
  const char *gname = basename_of(base);

  /* ---- 3. threads and memory ---- */
  bk_init(p);
  printf("%s | %s, threads = %d | n = %lld, nnz = %lld | u = %g, theta = %g, leaf cap m = %d\n",
         gname, bk_name(), bk_threads(), (long long)n, (long long)W.nnz, S.u, theta, leafcap);
  double *beta = malloc(n * sizeof *beta);
  double *P = malloc(W.nnz * sizeof *P);
  double *Fattr = malloc(2 * n * sizeof *Fattr);
  double *Frep = malloc(2 * n * sizeof *Frep);
  double *grad = malloc(2 * n * sizeof *grad);
  double Z = 0;
  tree_t T; tree_alloc(&T, n, leafcap);

  FILE *fc = csv ? fopen(csv, "a") : NULL;
  if (fc && ftell(fc) == 0)
    fprintf(fc, "graph,version,threads,n,nnz,leafcap,theta,rep,t_P,t_attr,"
                "t_build,t_sort,t_com,t_walk,t_grad,t_total,nnodes,nlevels,Z,"
                "err_beta,err_attr,err_rep,err_Z\n");

  /* ---- 4. the gradient, R times; every phase timed ---- */ // YΠΟΛΟΓΙΖΕΙ ΤΟ GRADIENT   και χρονομετρα καθε φαση
  double *tt[8];                       /* P attr build sort com walk grad total */
  for (int k = 0; k < 8; k++) tt[k] = malloc(reps * sizeof(double));
  for (int r = 0; r < reps; r++) {
    times_t t = {0};
    double t0 = wtime();
    bk_compute_P(&W, S.u, beta, P);                    double t1 = wtime();
    bk_attraction(&W, P, Y, Fattr);                    double t2 = wtime();
    bk_tree_build(&T, Y, &t);                          double t3 = wtime();
    bk_tree_com(&T);                                   double t4 = wtime();
    Z = bk_walk(&T, theta, Frep);                      double t5 = wtime();
    bk_gradient(n, Fattr, Frep, grad);                 double t6 = wtime();
    t.P = t1 - t0; t.attr = t2 - t1; t.build = t3 - t2;
    t.com = t4 - t3; t.walk = t5 - t4; t.grad = t6 - t5;
    double tot = t6 - t0;
    tt[0][r] = t.P; tt[1][r] = t.attr; tt[2][r] = t.build; tt[3][r] = t.build_sort;
    tt[4][r] = t.com; tt[5][r] = t.walk; tt[6][r] = t.grad; tt[7][r] = tot;

    if (fc) {                          /* one CSV line per repetition */
      double eb = NAN, ea = NAN, er = NAN, ez = NAN;
      if (have_ref) {
        eb = maxdiff(beta, R.beta, n); ea = relerr(Fattr, R.Fattr, 2 * n);
        er = relerr(Frep, R.Frep, 2 * n); ez = fabs(Z - R.Z) / R.Z;
      }
      fprintf(fc, "%s,%s,%d,%lld,%lld,%d,%g,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%lld,%d,%.15e,"
                  "%.3e,%.3e,%.3e,%.3e\n",
              gname, bk_name(), bk_threads(), (long long)n, (long long)W.nnz, leafcap,
              theta, r, t.P, t.attr, t.build, t.build_sort, t.com, t.walk, t.grad, tot,
              (long long)T.nnodes, T.nlevels, Z, eb, ea, er, ez);
    }
  }

  /* ---- 5. median times ---- */
  printf("  median of %d runs [s]: P %.4f | attr %.4f | build %.4f (sort %.4f) | com %.4f | walk %.4f | grad %.4f | total %.4f\n",
         reps, median(tt[0], reps), median(tt[1], reps), median(tt[2], reps),
         median(tt[3], reps), median(tt[4], reps), median(tt[5], reps),
         median(tt[6], reps), median(tt[7], reps));
  printf("  quadtree: %lld cells, %d levels; Z = %.10g\n",
         (long long)T.nnodes, T.nlevels, Z);

  /* ---- 6. compare with the reference ---- */
  if (have_ref) {
    double *gref = malloc(2 * n * sizeof *gref);
    for (int64_t k = 0; k < 2 * n; k++) gref[k] = 4.0 * (R.Fattr[k] - R.Frep[k]);
    printf("  vs reference: beta max|diff| %.2e | Fattr %.2e | Frep %.2e | Z %.2e | grad %.2e\n",
           maxdiff(beta, R.beta, n), relerr(Fattr, R.Fattr, 2 * n),
           relerr(Frep, R.Frep, 2 * n), fabs(Z - R.Z) / R.Z, relerr(grad, gref, 2 * n));
    free(gref);
  }

  /* ---- 7. optional experiments (experiments.c) ---- */
  if (do_stats) exp_stats(&T, theta);
  if (do_direct) exp_direct(gname, Y, n, W.nnz, Frep, Z, have_ref ? R.Frep : NULL,
                            have_ref ? R.Z : 0, reps, fc);
  if (xover) exp_crossover(gname, Y, n, leafcap, theta, reps, xover);

  /* ---- 8. save ---- */
  if (outdir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s-%s.mat", outdir, gname, bk_name());
    save(path, n, beta, Fattr, Frep, Z, grad);
    printf("  saved %s\n", path);
  }

  if (fc) fclose(fc);
  for (int k = 0; k < 8; k++) free(tt[k]);
  tree_free(&T);
  free(beta); free(P); free(Fattr); free(Frep); free(grad);
  free(W.rowptr); free(W.col); free(W.w); free(S.Y0);
  if (have_ref) { free(R.beta); free(R.Fattr); free(R.Frep); }
  return 0;
}
