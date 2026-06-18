#!/bin/bash
# run_scalability.sh — Scalability experiments for Table 5
# Run this script from the apo_drl/ directory after training at N=30.
# Estimated time: ~4 hours per N value on CPU; ~30 min on GPU.
# 
# Usage: cd apo_drl && bash run_scalability.sh
#
# Output CSVs will be saved to results/scalability/
# After completion, compute Table 5 stats with:
#   python3 src/get_stats.py --pattern "results/scalability/*.csv" --tail 300

set -e
OUTDIR="results/scalability"
EPISODES=3000
SEEDS="0 42 123"

echo "=== Scalability training: N=50, 100, 200 ==="
echo "Estimated time: 4-12 hours on CPU, 30-90 min on GPU"
echo ""

for N in 50 100 200; do
    echo "--- Training APO-DRL at N=$N ---"
    python3 src/train.py --method apo_drl --seeds $SEEDS --episodes $EPISODES \
        --devices $N --out_dir $OUTDIR/n${N}

    echo "--- Evaluating SA and AHP-TOPSIS at N=$N ---"
    python3 src/train.py --method static    --seeds $SEEDS --episodes 50 \
        --devices $N --out_dir $OUTDIR/n${N}
    python3 src/train.py --method ahp_topsis --seeds $SEEDS --episodes 50 \
        --devices $N --out_dir $OUTDIR/n${N}

    echo "--- N=$N complete ---"
    echo ""
done

echo "=== All scalability runs complete ==="
echo "Run: python3 src/get_stats.py --pattern 'results/scalability/**/*.csv' --tail 300"
