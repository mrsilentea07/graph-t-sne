"""Predict the time of each phase on coPapersDBLP from the runs on
k49_norm_10NN and com-Amazon only (results/raw/scaling.csv).

Models, fixed from the cost of each phase (not fitted to coPapersDBLP):
  P            ~ nnz     rate per nonzero of com-Amazon (unweighted, like coPapersDBLP)
  attraction   ~ nnz     mean rate per nonzero of k49 and com-Amazon
  build        ~ n       rate per point of com-Amazon (radix sort is O(n))
  centers      ~ n       rate per point of com-Amazon (cells ~ n / m)
  walk         ~ n log n rate of com-Amazon scaled by (n log n)
Writes results/prediction.csv.
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from csvutil import load, median_by

rows = [r for r in load('results/raw/scaling.csv')
        if r['graph'] in ('k49_norm_10NN', 'com-Amazon') and '-direct' not in r['version']]
N = {'k49_norm_10NN': (38547, 618158), 'com-Amazon': (334863, 1851744),
     'coPapersDBLP': (540486, 30491458)}
nC, eC = N['coPapersDBLP']
out = ['version,threads,phase,predicted']
for ver, p in [('seq', 1), ('omp', 1), ('omp', 8), ('omp', 16), ('cilk', 1), ('cilk', 8), ('cilk', 16)]:
    sel = [r for r in rows if r['version'] == ver and int(r['threads']) == p]
    med = lambda g, f: median_by([r for r in sel if r['graph'] == g], ['graph'], f)[(g,)]
    nA, eA = N['com-Amazon']; nK, eK = N['k49_norm_10NN']
    pred = {
        'P':     med('com-Amazon', 't_P') / eA * eC,
        'attr':  0.5 * (med('com-Amazon', 't_attr') / eA + med('k49_norm_10NN', 't_attr') / eK) * eC,
        'build': med('com-Amazon', 't_build') / nA * nC,
        'com':   med('com-Amazon', 't_com') / nA * nC,
        'walk':  med('com-Amazon', 't_walk') * (nC * math.log(nC)) / (nA * math.log(nA)),
    }
    for ph, v in pred.items():
        out.append(f'{ver},{p},{ph},{v:.6e}')
    print(f'{ver:4s} p={p:2d}: ' + ' | '.join(f'{k} {v:.4f}' for k, v in pred.items())
          + f' | total {sum(pred.values()):.4f}')
open('results/prediction.csv', 'w').write('\n'.join(out) + '\n')
print('wrote results/prediction.csv')
