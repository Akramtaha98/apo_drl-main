#!/bin/bash
cd "$(dirname "$0")"
OUTDIR="$HOME/Downloads/scalability/n100"
mkdir -p "$OUTDIR"
echo "=== N=100 training — saving to $OUTDIR ==="

python3 src/train.py --method apo_drl --seeds 0 42 123 --episodes 1000 --devices 100 --out_dir "$OUTDIR" --device cpu
python3 src/train.py --method static --seeds 0 42 123 --episodes 50 --devices 100 --out_dir "$OUTDIR" --device cpu
python3 src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 50 --devices 100 --out_dir "$OUTDIR" --device cpu

echo "=== N=100 DONE ===" && ls -lh "$OUTDIR"/*.csv
