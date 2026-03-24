#!/bin/bash
# Iterate over all major category directories and export SetFit models to ONNX.
# ONNX and tokenizer are written inside the best folder.
# If ONNX already exists, skip with a warning.

BASE_DIR="trainer/checkpoints/setfit/setfit-0324-0125"
RUNNER="uv run trainer/main.py setfit export-onnx"

SKIP_COUNT=0
EXPORT_COUNT=0

for major_dir in "$BASE_DIR"/*/; do
    major_name=$(basename "$major_dir")
    best_dir="${major_dir}best"
    onnx_path="${major_dir}best.onnx"

    if [ -d "$best_dir" ]; then
        if [ -f "$onnx_path/best.onnx" ]; then
            echo "[WARN] [$major_name] ONNX already exists, skipping: $onnx_path/best.onnx"
            SKIP_COUNT=$((SKIP_COUNT + 1))
        else
            echo "[INFO] [$major_name] Starting ONNX export..."
            $RUNNER -i "$best_dir" -o "$onnx_path"
            if [ $? -eq 0 ]; then
                EXPORT_COUNT=$((EXPORT_COUNT + 1))
                echo "[OK] [$major_name] Export succeeded"
            else
                echo "[FAIL] [$major_name] Export failed"
            fi
        fi
    else
        echo "[WARN] [$major_name] Unexpected directory structure (no best subdir), skipping: $best_dir"
    fi
    echo "---"
done

echo ""
echo "========== Done =========="
echo "Exported:  $EXPORT_COUNT"
echo "Skipped:   $SKIP_COUNT"
