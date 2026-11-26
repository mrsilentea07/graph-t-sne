/* tsne_cilk.c -- OpenCilk backend. Threads: CILK_NWORKERS=<p>.
 *
 * Phase           parallel construct
 * P (beta)        cilk_for over rows (work stealing balances degrees)
 * P (symmetrize)  cilk_for over rows
 * attraction      cilk_for over rows
 * build: bbox     cilk_for over blocks, block results combined serially
 *        keys     cilk_for
 *        sort     LSD radix sort: cilk_for histograms per block, scan, scatter
 *        levels   per level: cilk_for count of children, scan, cilk_for create
 * centers of mass per level, bottom-up: cilk_for over the cells
 * walk            cilk_for over sorted points, Z in a sum reducer
 */
#include "tsne.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROW_GRAIN   64
#define POINT_GRAIN 256

static double wt(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* sum reducer for doubles */
static void zero_d(void *v) { *(double *)v = 0.0; }
static void add_d(void *l, void *r) { *(double *)l += *(double *)r; }

const char *bk_name(void) { return "cilk"; }
void bk_init(int p) { (void)p; /* set CILK_NWORKERS in the environment */ }
int  bk_threads(void) { return __cilkrts_get_nworkers(); }

/* number of blocks for block-wise loops (bbox, radix sort) */
static int nblocks(int64_t n) {
  int64_t b = 4 * (int64_t)__cilkrts_get_nworkers();
  if (b > n / 1024) b = n / 1024;
  return b < 1 ? 1 : (int)b;
}

/* ---------------- P ---------------- */
void bk_compute_P(const csr_t *W, double u, double *beta, double *P) {
  const int64_t n = W->n;
  double *pc = malloc(W->nnz * sizeof *pc);
  #pragma cilk grainsize ROW_GRAIN
  cilk_for (int64_t i = 0; i < n; i++) {
    const int64_t k0 = W->rowptr[i];
    beta[i] = row_conditional(W->w + k0, W->rowptr[i + 1] - k0, u, pc + k0);
  }
  /* cilk_for ends with an implicit sync */
  const double s = 1.0 / (2.0 * (double)n);
  #pragma cilk grainsize ROW_GRAIN
  cilk_for (int64_t i = 0; i < n; i++)
    for (int64_t k = W->rowptr[i]; k < W->rowptr[i + 1]; k++)
      P[k] = (pc[k] + pc[find_entry(W, W->col[k], (int32_t)i)]) * s;
  free(pc);
}

/* ---------------- attraction ---------------- */
void bk_attraction(const csr_t *W, const double *P, const double *Y, double *F) {
  #pragma cilk grainsize ROW_GRAIN
  cilk_for (int64_t i = 0; i < W->n; i++) attr_row(W, P, Y, i, F);
}

/* ---------------- quadtree ---------------- */
static void radix_sort(uint64_t *key, int32_t *idx, int64_t n) {
  uint64_t *k2 = malloc(n * sizeof *k2);
  int32_t  *i2 = malloc(n * sizeof *i2);
  const int B = nblocks(n);
  int64_t (*cnt)[RADIX_SIZE] = malloc(B * sizeof *cnt);
  uint64_t *ka = key, *kb = k2; int32_t *ia = idx, *ib = i2;
  for (int pass = 0; pass < RADIX_PASSES; pass++) {
    const int sh = pass * RADIX_BITS;
    cilk_for (int q = 0; q < B; q++) {              /* histograms */
      const int64_t lo = n * q / B, hi = n * (q + 1) / B;
      memset(cnt[q], 0, sizeof cnt[q]);
      for (int64_t s = lo; s < hi; s++) cnt[q][(ka[s] >> sh) & (RADIX_SIZE - 1)]++;
    }
    int64_t off = 0;                                /* offsets (stable) */
    for (int d = 0; d < RADIX_SIZE; d++)
      for (int q = 0; q < B; q++) { int64_t c = cnt[q][d]; cnt[q][d] = off; off += c; }
    cilk_for (int q = 0; q < B; q++) {              /* scatter */
      const int64_t lo = n * q / B, hi = n * (q + 1) / B;
      for (int64_t s = lo; s < hi; s++) {
        int64_t pos = cnt[q][(ka[s] >> sh) & (RADIX_SIZE - 1)]++;
        kb[pos] = ka[s]; ib[pos] = ia[s];
      }
    }
    uint64_t *tk = ka; ka = kb; kb = tk; int32_t *ti = ia; ia = ib; ib = ti;
  }
  if (ka != key) {
    cilk_for (int64_t s = 0; s < n; s++) { key[s] = ka[s]; idx[s] = ia[s]; }
  }
  free(cnt); free(k2); free(i2);
}

void bk_tree_build(tree_t *T, const double *Y, times_t *t) {
  const double t0 = wt();
  const int64_t n = T->n;
  const int B = nblocks(n);
  double *bb = malloc(4 * B * sizeof *bb);
  cilk_for (int q = 0; q < B; q++) {                /* bounding box */
    const int64_t lo = n * q / B, hi = n * (q + 1) / B;
    double x0 = Y[2 * lo], x1 = x0, y0 = Y[2 * lo + 1], y1 = y0;
    for (int64_t i = lo + 1; i < hi; i++) {
      x0 = fmin(x0, Y[2 * i]); x1 = fmax(x1, Y[2 * i]);
      y0 = fmin(y0, Y[2 * i + 1]); y1 = fmax(y1, Y[2 * i + 1]);
    }
    bb[4 * q] = x0; bb[4 * q + 1] = x1; bb[4 * q + 2] = y0; bb[4 * q + 3] = y1;
  }
  double xmin = bb[0], xmax = bb[1], ymin = bb[2], ymax = bb[3];
  for (int q = 1; q < B; q++) {
    xmin = fmin(xmin, bb[4 * q]); xmax = fmax(xmax, bb[4 * q + 1]);
    ymin = fmin(ymin, bb[4 * q + 2]); ymax = fmax(ymax, bb[4 * q + 3]);
  }
  free(bb);
  set_bounding_square(T, xmin, xmax, ymin, ymax);
  cilk_for (int64_t i = 0; i < n; i++) {
    T->key[i] = morton_key(T, Y[2 * i], Y[2 * i + 1]);
    T->perm[i] = (int32_t)i;
  }
  radix_sort(T->key, T->perm, n);
  cilk_for (int64_t s = 0; s < n; s++) {
    T->pts[2 * s] = Y[2 * T->perm[s]]; T->pts[2 * s + 1] = Y[2 * T->perm[s] + 1];
  }
  t->build_sort = wt() - t0;

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
    #pragma cilk grainsize 64
    cilk_for (int64_t k = 0; k < m; k++) {          /* count children */
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
    #pragma cilk grainsize 64
    cilk_for (int64_t k = 0; k < m; k++)            /* create them */
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
  for (int l = T->nlevels - 1; l >= 0; l--) {       /* bottom-up */
    #pragma cilk grainsize 64
    cilk_for (int64_t k = T->lev[l]; k < T->lev[l + 1]; k++) com_cell(T, &T->node[k]);
  }
}

double bk_walk(const tree_t *T, double theta, double *Frep) {
  const int64_t n = T->n;
  const double th2 = theta * theta;
  double cilk_reducer(zero_d, add_d) Z = 0.0;
  #pragma cilk grainsize POINT_GRAIN
  cilk_for (int64_t s = 0; s < n; s++) {
    double fx, fy, z;
    walk_point(T, th2, s, &fx, &fy, &z);
    const int64_t i = T->perm[s];
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy;
    Z += z;
  }
  const double Zt = Z, iz = 1.0 / Zt;
  cilk_for (int64_t k = 0; k < 2 * n; k++) Frep[k] *= iz;
  return Zt;
}

double bk_direct(const double *Y, int64_t n, double *Frep) {
  double cilk_reducer(zero_d, add_d) Z = 0.0;
  cilk_for (int64_t i = 0; i < n; i++) {
    double fx, fy, z;
    direct_point(Y, n, i, &fx, &fy, &z);
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy; Z += z;
  }
  const double Zt = Z;
  cilk_for (int64_t k = 0; k < 2 * n; k++) Frep[k] /= Zt;
  return Zt;
}

void bk_gradient(int64_t n, const double *Fattr, const double *Frep, double *g) {
  cilk_for (int64_t k = 0; k < 2 * n; k++) g[k] = 4.0 * (Fattr[k] - Frep[k]);
}
