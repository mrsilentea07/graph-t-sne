/* overhead.c -- cost of starting a job / a barrier: our Pthreads pool vs OpenMP.
 *   gcc -O2 -fopenmp -pthread -o bin/overhead scripts/overhead.c -lm
 *   for p in 4 8 16; do bin/overhead $p; done
 *   OMP_WAIT_POLICY=passive bin/overhead 8     # OpenMP threads sleep at once too
 */
#include "../src/tsne_pthreads.c"
#include <omp.h>
void tree_reserve(tree_t *T, int64_t need) { (void)T; (void)need; }   /* not used */
static void empty(void *a, int t, int nt) { (void)a; (void)t; (void)nt; }
static void bar(void *a, int t, int nt) { (void)a; (void)t; (void)nt;
  for (int i = 0; i < 1000; i++) pthread_barrier_wait(&pool_bar); }
volatile int sink[64];
int main(int argc, char **argv) {
  int p = argc > 1 ? atoi(argv[1]) : 8;
  bk_init(p); omp_set_num_threads(p);
  const int R = 20000; double t0;
  for (int i = 0; i < 100; i++) run_job(empty, NULL);
  t0 = wt(); for (int i = 0; i < R; i++) run_job(empty, NULL); double job = (wt() - t0) / R;
  t0 = wt(); run_job(bar, NULL); double b = (wt() - t0) / 1000;
  for (int i = 0; i < 100; i++) { _Pragma("omp parallel") { sink[omp_get_thread_num()]++; } }
  t0 = wt(); for (int i = 0; i < R; i++) { _Pragma("omp parallel") { sink[omp_get_thread_num()]++; } }
  double reg = (wt() - t0) / R;
  t0 = wt(); _Pragma("omp parallel") { for (int i = 0; i < 1000; i++) { _Pragma("omp barrier") } }
  double ob = (wt() - t0) / 1000;
  printf("p=%2d  pthreads job %.1f us, barrier %.1f us | omp parallel %.1f us, barrier %.1f us\n",
         p, job * 1e6, b * 1e6, reg * 1e6, ob * 1e6);
  return 0;
}
