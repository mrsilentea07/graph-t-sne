/* experiments.h -- the extra measurements of the report, run by main.c
 * after the gradient (options --stats, --direct, --crossover).          */
#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H
#include "tsne.h"
#include <stdio.h>

/* small helpers, defined in main.c */
double wtime(void);                                     /* seconds        */
double median(double *x, int r);                        /* sorts x        */
double relerr(const double *x, const double *xr, int64_t m);  /* max|x-xr| / max|xr| */

/* --stats: cells visited per point in the walk, and leaf sizes */
void exp_stats(const tree_t *T, double theta);

/* --direct: repulsion by the direct O(n^2) sum, compared with Barnes-Hut
 * (Frep, Z) and with the reference (Fref, Zref; Fref = NULL if none).   */
void exp_direct(const char *gname, const double *Y, int64_t n, int64_t nnz,
                const double *Frep, double Z, const double *Fref, double Zref,
                int reps, FILE *csv);

/* --crossover: Barnes-Hut vs direct on evenly spaced subsets of the points */
void exp_crossover(const char *gname, const double *Y, int64_t n, int leafcap,
                   double theta, int reps, const char *path);

#endif
