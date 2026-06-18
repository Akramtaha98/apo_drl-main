#!/bin/bash
cd "$(dirname "$0")"
OUTDIR="$HOME/Downloads/scalability/n200"
mkdir -p "$OUTDIR"
echo "=== N=200 training — saving to $OUTDIR ==="

python3 src/train.py --method apo_drl --seeds 0 42 123 --episodes 600 --devices 200 --out_dir "$OUTDIR" --device cpu
python3 src/train.py --method static --seeds 0 42 123 --episodes 50 --devices 200 --out_dir "$OUTDIR" --device cpu
python3 src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 50 --devices 200 --out_dir "$OUTDIR" --device cpu

echo "=== N=200 DONE ===" && ls -lh "$OUTDIR"/*.csv
