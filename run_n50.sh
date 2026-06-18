#!/bin/bash
cd "$(dirname "$0")"
OUTDIR="$HOME/Downloads/scalability/n50"
mkdir -p "$OUTDIR"
echo "=== N=50 training — saving to $OUTDIR ==="

python3 src/train.py --method apo_drl --seeds 0 42 123 --episodes 1500 --devices 50 --out_dir "$OUTDIR" --device cpu
python3 src/train.py --method static --seeds 0 42 123 --episodes 50 --devices 50 --out_dir "$OUTDIR" --device cpu
python3 src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 50 --devices 50 --out_dir "$OUTDIR" --device cpu

echo "=== N=50 DONE ===" && ls -lh "$OUTDIR"/*.csv
