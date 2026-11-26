"""Tiny helpers to read the timing CSVs without pandas."""
import csv, statistics
from collections import defaultdict

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try: r[k] = float(v) if v not in ('', None) else float('nan')
                except ValueError: pass
            rows.append(r)
    return rows

def median_by(rows, keys, field):
    g = defaultdict(list)
    for r in rows: g[tuple(r[k] for k in keys)].append(r[field] if not callable(field) else field(r))
    return {k: statistics.median(v) for k, v in g.items()}
