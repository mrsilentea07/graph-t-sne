/* tsne_omp.c -- OpenMP backend.
 *
 * Phase           parallel construct
 * P (beta)        parallel for, schedule(runtime), default dynamic,64
 * P (symmetrize)  parallel for, schedule(runtime)  (row cost ~ degree)
 * attraction      parallel for, schedule(runtime)  (row cost ~ degree)
 * build: bbox     parallel for with min/max reductions
 *        keys     parallel for
 *        sort     LSD radix sort: per-thread histograms, scan, scatter
 *        levels   per level: parallel count of children, scan, parallel create
 * centers of mass per level, bottom-up: parallel for over the cells
 * walk            parallel for over sorted points, dynamic, reduction(+:Z)
 */
#include "tsne.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define ROW_CHUNK   64
#define POINT_CHUNK 256

const char *bk_name(void) { return "omp"; }
/* Row loops (P, attraction) use schedule(runtime): the default is
 * dynamic,64; OMP_SCHEDULE (e.g. "static", "guided") overrides it.     */
void bk_init(int p) {
  if (p > 0) omp_set_num_threads(p);
  if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_dynamic, ROW_CHUNK);
}
int  bk_threads(void) { return omp_get_max_threads(); }

/* ---------------- P ---------------- */
void bk_compute_P(const csr_t *W, double u, double *beta, double *P) {
  const int64_t n = W->n;
  double *pc = malloc(W->nnz * sizeof *pc);
  const double s = 1.0 / (2.0 * (double)n);
  #pragma omp parallel
  {
    #pragma omp for schedule(runtime)
    for (int64_t i = 0; i < n; i++) {
      const int64_t k0 = W->rowptr[i];
      beta[i] = row_conditional(W->w + k0, W->rowptr[i + 1] - k0, u, pc + k0);
    }
    /* implicit barrier: all p_{j|i} are ready before symmetrizing */
    #pragma omp for schedule(runtime)
    for (int64_t i = 0; i < n; i++)
      for (int64_t k = W->rowptr[i]; k < W->rowptr[i + 1]; k++)
        P[k] = (pc[k] + pc[find_entry(W, W->col[k], (int32_t)i)]) * s;
  }
  free(pc);
}

/* ---------------- attraction ---------------- */
void bk_attraction(const csr_t *W, const double *P, const double *Y, double *F) {
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < W->n; i++) attr_row(W, P, Y, i, F);
}

/* ---------------- quadtree ---------------- */
/* LSD radix sort of (key, idx), stable. Thread t owns the block
 * [t n / P, (t+1) n / P) in every pass.                                */
static void radix_sort(uint64_t *key, int32_t *idx, int64_t n) {
  uint64_t *k2 = malloc(n * sizeof *k2);
  int32_t  *i2 = malloc(n * sizeof *i2);
  const int P = omp_get_max_threads();
  int64_t (*cnt)[RADIX_SIZE] = malloc(P * sizeof *cnt);
  uint64_t *ka = key, *kb = k2; int32_t *ia = idx, *ib = i2;
  #pragma omp parallel num_threads(P)
  {
    const int t = omp_get_thread_num(), nt = omp_get_num_threads();
    const int64_t lo = n * t / nt, hi = n * (t + 1) / nt;
    for (int pass = 0; pass < RADIX_PASSES; pass++) {
      const int sh = pass * RADIX_BITS;
      memset(cnt[t], 0, sizeof cnt[t]);
      for (int64_t s = lo; s < hi; s++) cnt[t][(ka[s] >> sh) & (RADIX_SIZE - 1)]++;
      #pragma omp barrier
      #pragma omp single
      { /* offsets: digit-major, then thread order -> stable */
        int64_t off = 0;
        for (int d = 0; d < RADIX_SIZE; d++)
          for (int q = 0; q < nt; q++) { int64_t c = cnt[q][d]; cnt[q][d] = off; off += c; }
      } /* implicit barrier */
      for (int64_t s = lo; s < hi; s++) {
        int64_t pos = cnt[t][(ka[s] >> sh) & (RADIX_SIZE - 1)]++;
        kb[pos] = ka[s]; ib[pos] = ia[s];
      }
      #pragma omp barrier
      #pragma omp single
      { uint64_t *tk = ka; ka = kb; kb = tk; int32_t *ti = ia; ia = ib; ib = ti; }
    }
  }
  if (ka != key) {   /* odd number of passes: result is in the buffer */
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < n; s++) { key[s] = ka[s]; idx[s] = ia[s]; }
  }
  free(cnt); free(k2); free(i2);
}

void bk_tree_build(tree_t *T, const double *Y, times_t *t) {
  const double t0 = omp_get_wtime();
  const int64_t n = T->n;
  double xmin = Y[0], xmax = Y[0], ymin = Y[1], ymax = Y[1];
  #pragma omp parallel for schedule(static) \
          reduction(min:xmin, ymin) reduction(max:xmax, ymax)
  for (int64_t i = 0; i < n; i++) {
    xmin = fmin(xmin, Y[2 * i]); xmax = fmax(xmax, Y[2 * i]);
    ymin = fmin(ymin, Y[2 * i + 1]); ymax = fmax(ymax, Y[2 * i + 1]);
  }
  set_bounding_square(T, xmin, xmax, ymin, ymax);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    T->key[i] = morton_key(T, Y[2 * i], Y[2 * i + 1]);
    T->perm[i] = (int32_t)i;
  }
  radix_sort(T->key, T->perm, n);
  #pragma omp parallel for schedule(static)
  for (int64_t s = 0; s < n; s++) {
    T->pts[2 * s] = Y[2 * T->perm[s]]; T->pts[2 * s + 1] = Y[2 * T->perm[s] + 1];
  }
  t->build_sort = omp_get_wtime() - t0;

  node_t *root = &T->node[0];
  root->lo = 0; root->hi = (int32_t)n; root->child = -1; root->nchild = 0;
  root->diag2 = 2.0 * T->side * T->side;
  T->lev[0] = 0; T->lev[1] = 1;
  int l = 0;
  double side = T->side;
  while (T->lev[l + 1] > T->lev[l]) {
    const int64_t ls = T->lev[l], le = T->lev[l + 1], m = le - ls;
    int32_t *b = malloc(5 * m * sizeof *b);
    int64_t *first = malloc(m * sizeof *first);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t k = 0; k < m; k++) {               /* count children */
      node_t *c = &T->node[ls + k];
      int ne = must_split(T, c, l) ? split_cell(T, c, l, b + 5 * k) : 0;
      first[k] = ne;
      if (!ne) b[5 * k] = -1;
    }
    int64_t total = 0;                              /* exclusive scan */
    for (int64_t k = 0; k < m; k++) { int64_t c = first[k]; first[k] = total; total += c; }
    tree_reserve(T, le + total);
    side *= 0.5;
    const double cd2 = 2.0 * side * side;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t k = 0; k < m; k++)                 /* create them */
      if (b[5 * k] >= 0)
        make_children(T, &T->node[ls + k], b + 5 * k, le + first[k], cd2);
    free(b); free(first);
    T->lev[l + 2] = le + total;
    l++;
  }
  T->nlevels = l;
  T->nnodes = T->lev[l];
}

void bk_tree_com(tree_t *T) {
  #pragma omp parallel
  for (int l = T->nlevels - 1; l >= 0; l--) {       /* bottom-up */
    #pragma omp for schedule(dynamic, 64)
    for (int64_t k = T->lev[l]; k < T->lev[l + 1]; k++) com_cell(T, &T->node[k]);
    /* implicit barrier: level l done before level l-1 reads it */
  }
}

double bk_walk(const tree_t *T, double theta, double *Frep) {
  const int64_t n = T->n;
  const double th2 = theta * theta;
  double Z = 0.0;
  #pragma omp parallel for schedule(dynamic, POINT_CHUNK) reduction(+:Z)
  for (int64_t s = 0; s < n; s++) {
    double fx, fy, z;
    walk_point(T, th2, s, &fx, &fy, &z);
    const int64_t i = T->perm[s];
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy;
    Z += z;
  }
  const double iz = 1.0 / Z;
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < 2 * n; k++) Frep[k] *= iz;
  return Z;
}

double bk_direct(const double *Y, int64_t n, double *Frep) {
  double Z = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:Z)
  for (int64_t i = 0; i < n; i++) {
    double fx, fy, z;
    direct_point(Y, n, i, &fx, &fy, &z);
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy; Z += z;
  }
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < 2 * n; k++) Frep[k] /= Z;
  return Z;
}

void bk_gradient(int64_t n, const double *Fattr, const double *Frep, double *g) {
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < 2 * n; k++) g[k] = 4.0 * (Fattr[k] - Frep[k]);
}
