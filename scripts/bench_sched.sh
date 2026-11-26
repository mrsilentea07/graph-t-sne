#!/usr/bin/env bash
# OpenMP schedule of the row loops (P, attraction): static vs dynamic vs guided.
cd "$(dirname "$0")/.."
OUT=results/raw/schedule.csv; TMP=$(mktemp); rm -f "$OUT"
for g in k49_norm_10NN com-Amazon coPapersDBLP; do
  for s in static static,64 dynamic,64 dynamic,1024 guided; do
    for p in 8 16; do
      rm -f "$TMP"
      OMP_SCHEDULE=$s ./bin/tsne_omp data/$g -p $p -r 7 -c "$TMP" > /dev/null
      [ -f "$OUT" ] || sed -n '1s/^/schedule,/p' "$TMP" > "$OUT"
      sed -n "2,\$s/^/\"$s\",/p" "$TMP" >> "$OUT"
    done
  done
done
rm -f "$TMP"; echo "wrote $OUT"
