/* tsne_seq.c -- sequential backend: the reference for the parallel ones. */
#include "tsne.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wt(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

const char *bk_name(void) { return "seq"; }
void bk_init(int p) { (void)p; }
int  bk_threads(void) { return 1; }

/* ---------------- P ---------------- */
void bk_compute_P(const csr_t *W, double u, double *beta, double *P) {
  const int64_t n = W->n;
  double *pc = malloc(W->nnz * sizeof *pc);         /* p_{j|i}, row i */
  for (int64_t i = 0; i < n; i++) {
    const int64_t k0 = W->rowptr[i];
    beta[i] = row_conditional(W->w + k0, W->rowptr[i + 1] - k0, u, pc + k0);
  }
  const double s = 1.0 / (2.0 * (double)n);         /* symmetrize */
  for (int64_t i = 0; i < n; i++)
    for (int64_t k = W->rowptr[i]; k < W->rowptr[i + 1]; k++)
      P[k] = (pc[k] + pc[find_entry(W, W->col[k], (int32_t)i)]) * s;
  free(pc);
}

/* ---------------- attraction ---------------- */
void bk_attraction(const csr_t *W, const double *P, const double *Y, double *F) {
  for (int64_t i = 0; i < W->n; i++) attr_row(W, P, Y, i, F);
}

/* ---------------- quadtree ---------------- */
static void radix_sort(uint64_t *key, int32_t *idx, int64_t n) {
  uint64_t *k2 = malloc(n * sizeof *k2);
  int32_t  *i2 = malloc(n * sizeof *i2);
  for (int pass = 0; pass < RADIX_PASSES; pass++) {
    const int sh = pass * RADIX_BITS;
    int64_t cnt[RADIX_SIZE] = {0};
    for (int64_t s = 0; s < n; s++) cnt[(key[s] >> sh) & (RADIX_SIZE - 1)]++;
    int64_t off = 0;
    for (int d = 0; d < RADIX_SIZE; d++) { int64_t c = cnt[d]; cnt[d] = off; off += c; }
    for (int64_t s = 0; s < n; s++) {
      int64_t pos = cnt[(key[s] >> sh) & (RADIX_SIZE - 1)]++;
      k2[pos] = key[s]; i2[pos] = idx[s];
    }
    memcpy(key, k2, n * sizeof *key); memcpy(idx, i2, n * sizeof *idx);
  }
  free(k2); free(i2);
}

void bk_tree_build(tree_t *T, const double *Y, times_t *t) {
  const double t0 = wt();
  const int64_t n = T->n;
  double xmin = Y[0], xmax = Y[0], ymin = Y[1], ymax = Y[1];
  for (int64_t i = 1; i < n; i++) {
    xmin = fmin(xmin, Y[2 * i]); xmax = fmax(xmax, Y[2 * i]);
    ymin = fmin(ymin, Y[2 * i + 1]); ymax = fmax(ymax, Y[2 * i + 1]);
  }
  set_bounding_square(T, xmin, xmax, ymin, ymax);
  for (int64_t i = 0; i < n; i++) {
    T->key[i] = morton_key(T, Y[2 * i], Y[2 * i + 1]);
    T->perm[i] = (int32_t)i;
  }
  radix_sort(T->key, T->perm, n);
  for (int64_t s = 0; s < n; s++) {
    T->pts[2 * s] = Y[2 * T->perm[s]]; T->pts[2 * s + 1] = Y[2 * T->perm[s] + 1];
  }
  t->build_sort = wt() - t0;

  /* level by level: split every cell with more than m points */
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
    int64_t total = 0;
    for (int64_t k = 0; k < m; k++) {               /* count children */
      node_t *c = &T->node[ls + k];
      int ne = must_split(T, c, l) ? split_cell(T, c, l, b + 5 * k) : 0;
      first[k] = total; total += ne;
      if (!ne) b[5 * k] = -1;
    }
    tree_reserve(T, le + total);
    side *= 0.5;
    for (int64_t k = 0; k < m; k++)                 /* create them */
      if (b[5 * k] >= 0)
        make_children(T, &T->node[ls + k], b + 5 * k, le + first[k], 2.0 * side * side);
    free(b); free(first);
    T->lev[l + 2] = le + total;
    l++;
  }
  T->nlevels = l;
  T->nnodes = T->lev[l];
}

void bk_tree_com(tree_t *T) {
  for (int l = T->nlevels - 1; l >= 0; l--)          /* bottom-up */
    for (int64_t k = T->lev[l]; k < T->lev[l + 1]; k++) com_cell(T, &T->node[k]);
}

double bk_walk(const tree_t *T, double theta, double *Frep) {
  const int64_t n = T->n;
  const double th2 = theta * theta;
  double Z = 0.0;
  for (int64_t s = 0; s < n; s++) {
    double fx, fy, z;
    walk_point(T, th2, s, &fx, &fy, &z);
    const int64_t i = T->perm[s];
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy;
    Z += z;
  }
  const double iz = 1.0 / Z;
  for (int64_t k = 0; k < 2 * n; k++) Frep[k] *= iz;
  return Z;
}

double bk_direct(const double *Y, int64_t n, double *Frep) {
  double Z = 0.0;
  for (int64_t i = 0; i < n; i++) {
    double fx, fy, z;
    direct_point(Y, n, i, &fx, &fy, &z);
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy; Z += z;
  }
  for (int64_t k = 0; k < 2 * n; k++) Frep[k] /= Z;
  return Z;
}

void bk_gradient(int64_t n, const double *Fattr, const double *Frep, double *g) {
  for (int64_t k = 0; k < 2 * n; k++) g[k] = 4.0 * (Fattr[k] - Frep[k]);
}
