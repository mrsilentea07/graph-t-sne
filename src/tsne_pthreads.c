/* tsne_pthreads.c -- Pthreads backend. Threads: -p P (default: all cores).
 *
 * The P threads are created once (bk_init) and kept in a pool; the calling
 * thread is thread 0 and works too. A "job" is a function that every thread
 * runs (SPMD); run_job() starts it and returns when all threads finished it,
 * so the end of every job is a barrier. On top of this:
 *   parfor_dyn   loop split in chunks taken from a shared atomic counter
 *   parfor_stat  loop split in P equal contiguous blocks
 *
 * Phase           construct
 * P (beta)        parfor_dyn over rows, chunk 64 (row cost ~ degree)
 * P (symmetrize)  parfor_dyn over rows, chunk 64 (second job: all p_{j|i} ready)
 * attraction      parfor_dyn over rows, chunk 64
 * build: bbox     parfor_stat, per-thread min/max combined serially
 *        keys     parfor_stat
 *        sort     LSD radix sort in one job: per-thread histograms,
 *                 barrier, scan by thread 0, barrier, scatter, barrier
 *        levels   per level: parfor_dyn count of children, serial scan,
 *                 parfor_dyn create
 * centers of mass one job, bottom-up; per level a dynamic counter over the
 *                 cells and a pthread_barrier before the next level
 * walk            parfor_dyn over sorted points, chunk 256, per-thread Z
 */
#include "tsne.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ROW_CHUNK   64
#define POINT_CHUNK 256
#define CELL_CHUNK  64

static double wt(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ================================================================== */
/*                            thread pool                              */
/* ================================================================== */
typedef void (*job_fn)(void *arg, int t, int nt);

static int NT = 1;                       /* threads, including the caller */
static pthread_mutex_t pool_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  pool_start = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  pool_done  = PTHREAD_COND_INITIALIZER;
static job_fn   cur_job;
static void    *cur_arg;
static unsigned long generation;         /* +1 for every new job          */
static int      pending;                 /* workers still running the job */
static pthread_barrier_t pool_bar;       /* barrier inside a job (NT)     */

static void *worker(void *a) {
  const int t = (int)(intptr_t)a;
  unsigned long seen = 0;
  for (;;) {
    pthread_mutex_lock(&pool_mtx);
    while (generation == seen) pthread_cond_wait(&pool_start, &pool_mtx);
    seen = generation;
    job_fn f = cur_job; void *arg = cur_arg;
    pthread_mutex_unlock(&pool_mtx);

    f(arg, t, NT);

    pthread_mutex_lock(&pool_mtx);
    if (--pending == 0) pthread_cond_signal(&pool_done);
    pthread_mutex_unlock(&pool_mtx);
  }
  return NULL;
}

/* Every thread runs f(arg, t, NT); returns when all have finished.
 * The mutex hand-off also makes all writes of the job visible.         */
static void run_job(job_fn f, void *arg) {
  if (NT == 1) { f(arg, 0, 1); return; }
  pthread_mutex_lock(&pool_mtx);
  cur_job = f; cur_arg = arg; pending = NT - 1; generation++;
  pthread_cond_broadcast(&pool_start);
  pthread_mutex_unlock(&pool_mtx);
  f(arg, 0, NT);                         /* the caller is thread 0 */
  pthread_mutex_lock(&pool_mtx);
  while (pending > 0) pthread_cond_wait(&pool_done, &pool_mtx);
  pthread_mutex_unlock(&pool_mtx);
}

static inline int64_t blk(int64_t n, int t, int nt) { return n * t / nt; }

/* ---------------- parallel for ---------------- */
/* body(ctx, lo, hi, t): process items [lo, hi) as thread t */
typedef void (*range_fn)(void *ctx, int64_t lo, int64_t hi, int t);
typedef struct {
  range_fn body; void *ctx;
  int64_t lo, hi, chunk;
  _Atomic int64_t next;
} pfor_t;

static void pfor_dyn_job(void *a, int t, int nt) {
  pfor_t *p = a; (void)nt;
  for (;;) {
    const int64_t s = atomic_fetch_add_explicit(&p->next, p->chunk, memory_order_relaxed);
    if (s >= p->hi) break;
    p->body(p->ctx, s, s + p->chunk < p->hi ? s + p->chunk : p->hi, t);
  }
}
static void pfor_stat_job(void *a, int t, int nt) {
  pfor_t *p = a;
  const int64_t n = p->hi - p->lo;
  const int64_t lo = p->lo + blk(n, t, nt), hi = p->lo + blk(n, t + 1, nt);
  if (lo < hi) p->body(p->ctx, lo, hi, t);
}
static void parfor_dyn(int64_t lo, int64_t hi, int64_t chunk, range_fn body, void *ctx) {
  if (hi <= lo) return;
  pfor_t p = { body, ctx, lo, hi, chunk, lo };
  if (hi - lo <= chunk) { body(ctx, lo, hi, 0); return; }   /* not worth a job */
  run_job(pfor_dyn_job, &p);
}
static void parfor_stat(int64_t lo, int64_t hi, range_fn body, void *ctx) {
  if (hi <= lo) return;
  pfor_t p = { body, ctx, lo, hi, 0, lo };
  run_job(pfor_stat_job, &p);
}

/* one value per thread, on its own cache line (no false sharing) */
typedef struct { double v[4]; char pad[64 - 4 * sizeof(double)]; } slot_t;

/* ================================================================== */
const char *bk_name(void) { return "pthreads"; }
void bk_init(int p) {
  if (p <= 0) { long c = sysconf(_SC_NPROCESSORS_ONLN); p = c > 0 ? (int)c : 1; }
  NT = p;
  pthread_barrier_init(&pool_bar, NULL, NT);
  for (int t = 1; t < NT; t++) {
    pthread_t th;
    if (pthread_create(&th, NULL, worker, (void *)(intptr_t)t)) {
      fprintf(stderr, "pthread_create failed\n"); exit(1);
    }
    pthread_detach(th);                  /* lives until the process ends */
  }
}
int bk_threads(void) { return NT; }

/* ---------------- P ---------------- */
typedef struct {
  const csr_t *W; double u, s; double *beta, *P, *pc;
} p_ctx;

static void beta_body(void *c, int64_t lo, int64_t hi, int t) {
  p_ctx *x = c; (void)t;
  for (int64_t i = lo; i < hi; i++) {
    const int64_t k0 = x->W->rowptr[i];
    x->beta[i] = row_conditional(x->W->w + k0, x->W->rowptr[i + 1] - k0, x->u, x->pc + k0);
  }
}
static void sym_body(void *c, int64_t lo, int64_t hi, int t) {
  p_ctx *x = c; const csr_t *W = x->W; (void)t;
  for (int64_t i = lo; i < hi; i++)
    for (int64_t k = W->rowptr[i]; k < W->rowptr[i + 1]; k++)
      x->P[k] = (x->pc[k] + x->pc[find_entry(W, W->col[k], (int32_t)i)]) * x->s;
}

void bk_compute_P(const csr_t *W, double u, double *beta, double *P) {
  const int64_t n = W->n;
  p_ctx x = { W, u, 1.0 / (2.0 * (double)n), beta, P, malloc(W->nnz * sizeof(double)) };
  parfor_dyn(0, n, ROW_CHUNK, beta_body, &x);
  /* end of job = barrier: all p_{j|i} are ready before symmetrizing */
  parfor_dyn(0, n, ROW_CHUNK, sym_body, &x);
  free(x.pc);
}

/* ---------------- attraction ---------------- */
typedef struct { const csr_t *W; const double *P, *Y; double *F; } a_ctx;
static void attr_body(void *c, int64_t lo, int64_t hi, int t) {
  a_ctx *x = c; (void)t;
  for (int64_t i = lo; i < hi; i++) attr_row(x->W, x->P, x->Y, i, x->F);
}
void bk_attraction(const csr_t *W, const double *P, const double *Y, double *F) {
  a_ctx x = { W, P, Y, F };
  parfor_dyn(0, W->n, ROW_CHUNK, attr_body, &x);
}

/* ---------------- quadtree ---------------- */
typedef struct { uint64_t *key; int32_t *idx; int64_t n; uint64_t *k2; int32_t *i2;
                 int64_t (*cnt)[RADIX_SIZE]; } sort_ctx;

/* LSD radix sort of (key, idx), stable. Thread t owns the block
 * [t n / P, (t+1) n / P) in every pass; offsets are digit-major, then
 * thread order, so equal digits keep their order.                      */
static void sort_job(void *a, int t, int nt) {
  sort_ctx *x = a;
  const int64_t lo = blk(x->n, t, nt), hi = blk(x->n, t + 1, nt);
  uint64_t *ka = x->key, *kb = x->k2; int32_t *ia = x->idx, *ib = x->i2;
  int64_t *my = x->cnt[t];
  for (int pass = 0; pass < RADIX_PASSES; pass++) {
    const int sh = pass * RADIX_BITS;
    memset(my, 0, RADIX_SIZE * sizeof *my);
    for (int64_t s = lo; s < hi; s++) my[(ka[s] >> sh) & (RADIX_SIZE - 1)]++;
    pthread_barrier_wait(&pool_bar);
    if (t == 0) {
      int64_t off = 0;
      for (int d = 0; d < RADIX_SIZE; d++)
        for (int q = 0; q < nt; q++) { int64_t c = x->cnt[q][d]; x->cnt[q][d] = off; off += c; }
    }
    pthread_barrier_wait(&pool_bar);
    for (int64_t s = lo; s < hi; s++) {
      const int64_t pos = my[(ka[s] >> sh) & (RADIX_SIZE - 1)]++;
      kb[pos] = ka[s]; ib[pos] = ia[s];
    }
    pthread_barrier_wait(&pool_bar);     /* all writes done before next pass */
    uint64_t *tk = ka; ka = kb; kb = tk; int32_t *ti = ia; ia = ib; ib = ti;
  }
  if (ka != x->key)                      /* odd number of passes */
    for (int64_t s = lo; s < hi; s++) { x->key[s] = ka[s]; x->idx[s] = ia[s]; }
}

static void radix_sort(uint64_t *key, int32_t *idx, int64_t n) {
  sort_ctx x = { key, idx, n, malloc(n * sizeof(uint64_t)), malloc(n * sizeof(int32_t)),
                 malloc(NT * sizeof *x.cnt) };
  run_job(sort_job, &x);
  free(x.cnt); free(x.k2); free(x.i2);
}

typedef struct {
  tree_t *T; const double *Y; slot_t *mm;            /* bbox, keys, permute */
  int l; int64_t ls; int32_t *b; int64_t *first; double cd2;   /* levels */
} b_ctx;

static void bbox_body(void *c, int64_t lo, int64_t hi, int t) {
  b_ctx *x = c; const double *Y = x->Y;
  double xmin = Y[2 * lo], xmax = xmin, ymin = Y[2 * lo + 1], ymax = ymin;
  for (int64_t i = lo + 1; i < hi; i++) {
    xmin = fmin(xmin, Y[2 * i]); xmax = fmax(xmax, Y[2 * i]);
    ymin = fmin(ymin, Y[2 * i + 1]); ymax = fmax(ymax, Y[2 * i + 1]);
  }
  x->mm[t].v[0] = xmin; x->mm[t].v[1] = xmax; x->mm[t].v[2] = ymin; x->mm[t].v[3] = ymax;
}
static void key_body(void *c, int64_t lo, int64_t hi, int t) {
  b_ctx *x = c; tree_t *T = x->T; (void)t;
  for (int64_t i = lo; i < hi; i++) {
    T->key[i] = morton_key(T, x->Y[2 * i], x->Y[2 * i + 1]);
    T->perm[i] = (int32_t)i;
  }
}
static void perm_body(void *c, int64_t lo, int64_t hi, int t) {
  b_ctx *x = c; tree_t *T = x->T; (void)t;
  for (int64_t s = lo; s < hi; s++) {
    T->pts[2 * s] = x->Y[2 * T->perm[s]]; T->pts[2 * s + 1] = x->Y[2 * T->perm[s] + 1];
  }
}
static void count_body(void *c, int64_t lo, int64_t hi, int t) {
  b_ctx *x = c; tree_t *T = x->T; (void)t;
  for (int64_t k = lo; k < hi; k++) {
    node_t *nd = &T->node[x->ls + k];
    int ne = must_split(T, nd, x->l) ? split_cell(T, nd, x->l, x->b + 5 * k) : 0;
    x->first[k] = ne;
    if (!ne) x->b[5 * k] = -1;
  }
}
static void create_body(void *c, int64_t lo, int64_t hi, int t) {
  b_ctx *x = c; tree_t *T = x->T; (void)t;
  for (int64_t k = lo; k < hi; k++)
    if (x->b[5 * k] >= 0)
      make_children(T, &T->node[x->ls + k], x->b + 5 * k,
                    T->lev[x->l + 1] + x->first[k], x->cd2);
}

void bk_tree_build(tree_t *T, const double *Y, times_t *t) {
  const double t0 = wt();
  const int64_t n = T->n;
  b_ctx x = { .T = T, .Y = Y, .mm = malloc(NT * sizeof(slot_t)) };
  for (int q = 0; q < NT; q++) {         /* threads with an empty block */
    x.mm[q].v[0] = x.mm[q].v[1] = Y[0]; x.mm[q].v[2] = x.mm[q].v[3] = Y[1];
  }
  parfor_stat(0, n, bbox_body, &x);
  double xmin = Y[0], xmax = Y[0], ymin = Y[1], ymax = Y[1];
  for (int q = 0; q < NT; q++) {
    xmin = fmin(xmin, x.mm[q].v[0]); xmax = fmax(xmax, x.mm[q].v[1]);
    ymin = fmin(ymin, x.mm[q].v[2]); ymax = fmax(ymax, x.mm[q].v[3]);
  }
  free(x.mm);
  set_bounding_square(T, xmin, xmax, ymin, ymax);
  parfor_stat(0, n, key_body, &x);
  radix_sort(T->key, T->perm, n);
  parfor_stat(0, n, perm_body, &x);
  t->build_sort = wt() - t0;

  node_t *root = &T->node[0];
  root->lo = 0; root->hi = (int32_t)n; root->child = -1; root->nchild = 0;
  root->diag2 = 2.0 * T->side * T->side;
  T->lev[0] = 0; T->lev[1] = 1;
  int l = 0;
  double side = T->side;
  while (T->lev[l + 1] > T->lev[l]) {
    const int64_t ls = T->lev[l], le = T->lev[l + 1], m = le - ls;
    x.l = l; x.ls = ls;
    x.b = malloc(5 * m * sizeof *x.b);
    x.first = malloc(m * sizeof *x.first);
    parfor_dyn(0, m, CELL_CHUNK, count_body, &x);    /* count children */
    int64_t total = 0;                                /* exclusive scan */
    for (int64_t k = 0; k < m; k++) { int64_t c = x.first[k]; x.first[k] = total; total += c; }
    tree_reserve(T, le + total);         /* may move T->node: serial, between jobs */
    side *= 0.5;
    x.cd2 = 2.0 * side * side;
    parfor_dyn(0, m, CELL_CHUNK, create_body, &x);   /* create them */
    free(x.b); free(x.first);
    T->lev[l + 2] = le + total;
    l++;
  }
  T->nlevels = l;
  T->nnodes = T->lev[l];
}

/* centers of mass: one job for all levels, a barrier between levels */
typedef struct { tree_t *T; _Atomic int64_t *next; } com_ctx;
static void com_job(void *a, int t, int nt) {
  com_ctx *x = a; tree_t *T = x->T; (void)t; (void)nt;
  for (int l = T->nlevels - 1; l >= 0; l--) {        /* bottom-up */
    const int64_t hi = T->lev[l + 1];
    for (;;) {
      const int64_t s = atomic_fetch_add_explicit(&x->next[l], CELL_CHUNK, memory_order_relaxed);
      if (s >= hi) break;
      const int64_t e = s + CELL_CHUNK < hi ? s + CELL_CHUNK : hi;
      for (int64_t k = s; k < e; k++) com_cell(T, &T->node[k]);
    }
    pthread_barrier_wait(&pool_bar);     /* level l done before l-1 reads it */
  }
}
void bk_tree_com(tree_t *T) {
  com_ctx x = { T, malloc((T->nlevels + 1) * sizeof(_Atomic int64_t)) };
  for (int l = 0; l < T->nlevels; l++) atomic_init(&x.next[l], T->lev[l]);
  run_job(com_job, &x);
  free(x.next);
}

/* ---------------- repulsion ---------------- */
typedef struct { const tree_t *T; const double *Y; int64_t n; double th2, scale;
                 double *Frep; slot_t *z; } w_ctx;

static void walk_body(void *c, int64_t lo, int64_t hi, int t) {
  w_ctx *x = c; double Z = 0.0;
  for (int64_t s = lo; s < hi; s++) {
    double fx, fy, z;
    walk_point(x->T, x->th2, s, &fx, &fy, &z);
    const int64_t i = x->T->perm[s];
    x->Frep[2 * i] = fx; x->Frep[2 * i + 1] = fy;
    Z += z;
  }
  x->z[t].v[0] += Z;
}
static void direct_body(void *c, int64_t lo, int64_t hi, int t) {
  w_ctx *x = c; double Z = 0.0;
  for (int64_t i = lo; i < hi; i++) {
    double fx, fy, z;
    direct_point(x->Y, x->n, i, &fx, &fy, &z);
    x->Frep[2 * i] = fx; x->Frep[2 * i + 1] = fy; Z += z;
  }
  x->z[t].v[0] += Z;
}
static void scale_body(void *c, int64_t lo, int64_t hi, int t) {
  w_ctx *x = c; (void)t;
  for (int64_t k = lo; k < hi; k++) x->Frep[k] *= x->scale;
}
static void div_body(void *c, int64_t lo, int64_t hi, int t) {
  w_ctx *x = c; (void)t;
  for (int64_t k = lo; k < hi; k++) x->Frep[k] /= x->scale;
}
static double sum_slots(const slot_t *z) {
  double Z = 0.0;
  for (int q = 0; q < NT; q++) Z += z[q].v[0];
  return Z;
}

double bk_walk(const tree_t *T, double theta, double *Frep) {
  const int64_t n = T->n;
  w_ctx x = { .T = T, .n = n, .th2 = theta * theta, .Frep = Frep,
              .z = calloc(NT, sizeof(slot_t)) };
  parfor_dyn(0, n, POINT_CHUNK, walk_body, &x);
  const double Z = sum_slots(x.z);
  free(x.z);
  x.scale = 1.0 / Z;
  parfor_stat(0, 2 * n, scale_body, &x);
  return Z;
}

double bk_direct(const double *Y, int64_t n, double *Frep) {
  w_ctx x = { .Y = Y, .n = n, .Frep = Frep, .z = calloc(NT, sizeof(slot_t)) };
  parfor_stat(0, n, direct_body, &x);
  const double Z = sum_slots(x.z);
  free(x.z);
  x.scale = Z;
  parfor_stat(0, 2 * n, div_body, &x);
  return Z;
}

/* ---------------- gradient ---------------- */
typedef struct { const double *Fa, *Fr; double *g; } g_ctx;
static void grad_body(void *c, int64_t lo, int64_t hi, int t) {
  g_ctx *x = c; (void)t;
  for (int64_t k = lo; k < hi; k++) x->g[k] = 4.0 * (x->Fa[k] - x->Fr[k]);
}
void bk_gradient(int64_t n, const double *Fattr, const double *Frep, double *g) {
  g_ctx x = { Fattr, Frep, g };
  parfor_stat(0, 2 * n, grad_body, &x);
}
