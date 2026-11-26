/* tsne.h -- one gradient of graph t-SNE (Barnes-Hut), shared definitions.
 *
 * Layout of the code:
 *   tsne.h       data structures, the backend interface, and the per-row /
 *                per-point kernels (static inline) that every version uses,
 *                so all versions do exactly the same arithmetic per item.
 *   main.c       driver: read data, time each phase, check, save results.
 *   tsne_seq.c   sequential backend.
 *   tsne_omp.c   OpenMP backend.
 *   tsne_cilk.c  OpenCilk backend.
 *   tsne_pthreads.c  Pthreads backend (thread pool).
 * Each backend implements the parallel loops of every phase in its own model.
 */
#ifndef TSNE_H
#define TSNE_H
#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Sparse matrix in CSR form. W is symmetric, so the CSC matrix read
 * from the .mat file is also its CSR form (row i == column i).        */
typedef struct {
  int64_t n, nnz;
  int64_t *rowptr;   /* n + 1                         */
  int32_t *col;      /* nnz, sorted within each row   */
  double  *w;        /* nnz, affinities w_ij > 0      */
} csr_t;

/* ------------------------------------------------------------------ */
/* Adaptive quadtree, stored level by level.
 * The points are sorted along a Morton (Z-order) curve, so the points of
 * every cell are a contiguous range [lo, hi) of the sorted arrays.     */
#define TREE_MAXDEPTH 20               /* bits per coordinate in the key */

typedef struct {
  double  cx, cy;      /* center of mass                                 */
  double  diag2;       /* (cell diagonal)^2 = 2 * side^2                 */
  int32_t lo, hi;      /* points of the cell: sorted indices [lo, hi)    */
  int32_t child;       /* index of first child, -1 for a leaf            */
  int32_t nchild;      /* number of (nonempty) children, contiguous      */
} node_t;

typedef struct {
  int64_t  n;          /* number of points                               */
  int      leafcap;    /* m: split a cell while it holds more than m     */
  double   x0, y0, side;  /* bounding square of the root                 */
  uint64_t *key;       /* n, sorted Morton keys                          */
  int32_t  *perm;      /* n, perm[s] = original index of sorted point s  */
  double   *pts;       /* 2n, sorted points (x, y interleaved)           */
  node_t   *node;      /* all cells, level by level                      */
  int64_t   nnodes, cap;
  int       nlevels;
  int64_t   lev[TREE_MAXDEPTH + 3];  /* level l: nodes lev[l] .. lev[l+1]-1 */
} tree_t;

/* Phase timings, seconds. */
typedef struct {
  double P, attr, build, com, walk, grad;
  double build_sort;   /* part of build: bbox + keys + radix sort + permute */
} times_t;

/* ------------------------------------------------------------------ */
/* Backend interface: one implementation per programming model.        */
const char *bk_name(void);
void   bk_init(int nthreads);       /* p = requested threads (ignored by seq) */
int    bk_threads(void);            /* threads actually used                  */
void   bk_compute_P(const csr_t *W, double u, double *beta, double *P);
void   bk_attraction(const csr_t *W, const double *P, const double *Y, double *F);
void   bk_tree_build(tree_t *T, const double *Y, times_t *t);
void   bk_tree_com(tree_t *T);
double bk_walk(const tree_t *T, double theta, double *Frep);  /* returns Z */
double bk_direct(const double *Y, int64_t n, double *Frep);   /* returns Z */
void   bk_gradient(int64_t n, const double *Fattr, const double *Frep, double *g);

/* Tree memory helpers (main.c) */
void tree_alloc(tree_t *T, int64_t n, int leafcap);
void tree_free(tree_t *T);
void tree_reserve(tree_t *T, int64_t need);   /* grow node array (serial) */

/* ================================================================== */
/*                      Kernels shared by all versions                 */
/* ================================================================== */

#define BISECT_TOL   1e-5
#define BISECT_MAXIT 200

/* Row i of P: conditional probabilities p_{j|i} over the neighbors,
 * with beta_i found by bisection so that the entropy is log(min(u, d)).
 * w: the d affinities of the row; p: output (d values). Returns beta_i
 * (0 when the row is uniform).                                         */
static inline double row_conditional(const double *w, int64_t d, double u,
                                     double *p) {
  if (d <= 0) return 0.0;
  double wmax = w[0], wmin = w[0];
  for (int64_t k = 1; k < d; k++) {
    if (w[k] > wmax) wmax = w[k];
    if (w[k] < wmin) wmin = w[k];
  }
  if (u >= (double)d || wmin == wmax) {           /* uniform row */
    for (int64_t k = 0; k < d; k++) p[k] = 1.0 / (double)d;
    return 0.0;
  }
  /* p[k] temporarily holds l_k = log(w_k / wmax) <= 0, so that
   * (w_k / wmax)^beta = exp(beta * l_k) never overflows.               */
  for (int64_t k = 0; k < d; k++) p[k] = log(w[k] / wmax);
  const double target = log(u);                   /* u < d here */
  double beta = 1.0, lo = 0.0, hi = INFINITY;
  for (int it = 0; it < BISECT_MAXIT; it++) {
    double S = 0.0, SL = 0.0;
    for (int64_t k = 0; k < d; k++) {
      double e = exp(beta * p[k]);
      S += e; SL += e * p[k];
    }
    /* H = -sum p log p, with p = e/S:  H = log S - beta * SL / S */
    double H = log(S) - beta * SL / S;
    if (fabs(H - target) < BISECT_TOL) break;
    if (H > target) {                 /* too flat: sharpen, beta up   */
      lo = beta;
      beta = isinf(hi) ? 2.0 * beta : 0.5 * (lo + hi);
    } else {                          /* too sharp: beta down         */
      hi = beta;
      beta = 0.5 * (lo + hi);
    }
  }
  double S = 0.0;
  for (int64_t k = 0; k < d; k++) S += exp(beta * p[k]);
  for (int64_t k = 0; k < d; k++) p[k] = exp(beta * p[k]) / S;
  return beta;
}

/* Position of entry (j, i) in row j (binary search; rows are sorted). */
static inline int64_t find_entry(const csr_t *W, int64_t j, int32_t i) {
  int64_t lo = W->rowptr[j], hi = W->rowptr[j + 1] - 1;
  while (lo <= hi) {
    int64_t mid = (lo + hi) >> 1;
    int32_t c = W->col[mid];
    if (c == i) return mid;
    if (c < i) lo = mid + 1; else hi = mid - 1;
  }
  return -1;   /* cannot happen for a symmetric pattern */
}

/* Row i of the attraction: sum_j p_ij (1 + d_ij^2)^-1 (y_i - y_j).
 * The entries of A = P .* (1 + D^2)^-1 are computed on the fly.        */
static inline void attr_row(const csr_t *W, const double *P, const double *Y,
                            int64_t i, double *F) {
  const double xi = Y[2 * i], yi = Y[2 * i + 1];
  double fx = 0.0, fy = 0.0;
  for (int64_t k = W->rowptr[i]; k < W->rowptr[i + 1]; k++) {
    const int64_t j = W->col[k];
    const double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
    const double a = P[k] / (1.0 + dx * dx + dy * dy);
    fx += a * dx; fy += a * dy;
  }
  F[2 * i] = fx; F[2 * i + 1] = fy;
}

/* Barnes-Hut walk for sorted point s. Accumulates the unnormalized
 * repulsion (fx, fy) and the contribution of point s to Z.
 * Returns the number of cells visited (for statistics).               */
#define WALK_STACK 128
static inline int64_t walk_point(const tree_t *T, double theta2, int64_t s,
                                 double *fx_, double *fy_, double *z_) {
  const node_t *nd = T->node;
  const double *pts = T->pts;
  const double xi = pts[2 * s], yi = pts[2 * s + 1];
  double fx = 0.0, fy = 0.0, z = 0.0;
  int32_t stack[WALK_STACK];
  int sp = 0;
  int64_t visited = 0;
  stack[sp++] = 0;                                  /* root */
  while (sp > 0) {
    const node_t *c = &nd[stack[--sp]];
    visited++;
    const double dx = xi - c->cx, dy = yi - c->cy;
    const double d2 = dx * dx + dy * dy;
    /* r_cell / ||y_i - y_cell|| < theta, squared, with r = diagonal */
    if (c->diag2 < theta2 * d2) {                   /* summarize cell */
      const double q = 1.0 / (1.0 + d2), N = (double)(c->hi - c->lo);
      z  += N * q;
      fx += N * q * q * dx;
      fy += N * q * q * dy;
    } else if (c->child < 0) {                      /* leaf: one by one */
      for (int32_t t = c->lo; t < c->hi; t++) {
        if (t == s) continue;                       /* no self-interaction */
        const double ex = xi - pts[2 * t], ey = yi - pts[2 * t + 1];
        const double q = 1.0 / (1.0 + ex * ex + ey * ey);
        z  += q;
        fx += q * q * ex;
        fy += q * q * ey;
      }
    } else {                                        /* open the cell */
      for (int32_t k = 0; k < c->nchild; k++) stack[sp++] = c->child + k;
    }
  }
  *fx_ = fx; *fy_ = fy; *z_ = z;
  return visited;
}

/* Exact O(n) repulsion sum for point i (direct method). */
static inline void direct_point(const double *Y, int64_t n, int64_t i,
                                double *fx_, double *fy_, double *z_) {
  const double xi = Y[2 * i], yi = Y[2 * i + 1];
  double fx = 0.0, fy = 0.0, z = 0.0;
  for (int64_t j = 0; j < n; j++) {
    const double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
    const double q = 1.0 / (1.0 + dx * dx + dy * dy);
    const double m = (j == i) ? 0.0 : 1.0;          /* branch-free skip */
    z  += m * q;
    fx += m * q * q * dx;
    fy += m * q * q * dy;
  }
  *fx_ = fx; *fy_ = fy; *z_ = z;
}

/* ---- quadtree helpers ---- */

/* Spread the low 20 bits of x so there is a zero bit between them. */
static inline uint64_t spread_bits(uint64_t x) {
  x &= 0xFFFFFull;
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
  x = (x | (x << 8))  & 0x00FF00FF00FF00FFull;
  x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0Full;
  x = (x | (x << 2))  & 0x3333333333333333ull;
  x = (x | (x << 1))  & 0x5555555555555555ull;
  return x;
}

/* Morton key of a point inside the bounding square: 2 bits per level,
 * the top level in the highest bits.                                   */
static inline uint64_t morton_key(const tree_t *T, double x, double y) {
  const double scale = (double)(1u << TREE_MAXDEPTH) / T->side;
  int64_t qx = (int64_t)((x - T->x0) * scale);
  int64_t qy = (int64_t)((y - T->y0) * scale);
  const int64_t qmax = (1 << TREE_MAXDEPTH) - 1;
  if (qx < 0) qx = 0;
  if (qx > qmax) qx = qmax;
  if (qy < 0) qy = 0;
  if (qy > qmax) qy = qmax;
  return (spread_bits((uint64_t)qx) << 1) | spread_bits((uint64_t)qy);
}

/* First sorted index in [lo, hi) whose key is >= k. */
static inline int32_t lower_bound_key(const uint64_t *key, int32_t lo,
                                      int32_t hi, uint64_t k) {
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    if (key[mid] < k) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* Split cell c (at level l) into its four quadrants: b[0..4] are the
 * boundaries of the four children in the sorted order. Returns the
 * number of nonempty children.                                         */
static inline int split_cell(const tree_t *T, const node_t *c, int l,
                             int32_t b[5]) {
  const int shift = 2 * (TREE_MAXDEPTH - 1 - l);    /* digit of level l+1 */
  const uint64_t prefix = (T->key[c->lo] >> (shift + 2)) << (shift + 2);
  b[0] = c->lo; b[4] = c->hi;
  for (int q = 1; q < 4; q++)
    b[q] = lower_bound_key(T->key, b[q - 1], c->hi,
                           prefix | ((uint64_t)q << shift));
  int ne = 0;
  for (int q = 0; q < 4; q++) ne += (b[q + 1] > b[q]);
  return ne;
}

/* Does cell c at level l need to be split? */
static inline int must_split(const tree_t *T, const node_t *c, int l) {
  return (c->hi - c->lo) > T->leafcap && l < TREE_MAXDEPTH;
}

/* Write the nonempty children of c (boundaries b) starting at node[first]. */
static inline void make_children(tree_t *T, node_t *c, const int32_t b[5],
                                 int64_t first, double child_diag2) {
  int k = 0;
  for (int q = 0; q < 4; q++) {
    if (b[q + 1] == b[q]) continue;
    node_t *ch = &T->node[first + k++];
    ch->lo = b[q]; ch->hi = b[q + 1];
    ch->child = -1; ch->nchild = 0;
    ch->diag2 = child_diag2;
    ch->cx = ch->cy = 0.0;
  }
  c->child = (int32_t)first;
  c->nchild = k;
}

/* Center of mass of cell c: from its points (leaf) or its children. */
static inline void com_cell(tree_t *T, node_t *c) {
  double sx = 0.0, sy = 0.0;
  if (c->child < 0) {
    for (int32_t t = c->lo; t < c->hi; t++) {
      sx += T->pts[2 * t]; sy += T->pts[2 * t + 1];
    }
  } else {
    for (int32_t k = 0; k < c->nchild; k++) {
      const node_t *ch = &T->node[c->child + k];
      const double N = (double)(ch->hi - ch->lo);
      sx += N * ch->cx; sy += N * ch->cy;
    }
  }
  const double N = (double)(c->hi - c->lo);
  c->cx = sx / N; c->cy = sy / N;
}

/* Bounding square from the bounding box (slightly enlarged so that no
 * point falls exactly on the upper edge).                              */
static inline void set_bounding_square(tree_t *T, double xmin, double xmax,
                                       double ymin, double ymax) {
  double side = fmax(xmax - xmin, ymax - ymin);
  if (side <= 0) side = 1.0;
  side *= 1.0 + 1e-9;
  T->x0 = xmin; T->y0 = ymin; T->side = side;
}

#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)
#define RADIX_PASSES ((2 * TREE_MAXDEPTH + RADIX_BITS - 1) / RADIX_BITS)

#endif
