#!/bin/bash
# Export SetFit / Supervised ONNX models by date folder.
# Scans both setfit/ and sub/ checkpoint trees, finds setfit-{DATE_TIME} and supervised-{DATE_TIME}
# folders, exports ONNX from best/ subfolder to best.onnx in the same folder.
#
# Usage: ./scripts/export_setfit_onnx_by_date.sh <DATE_TIME>
# Example: ./scripts/export_setfit_onnx_by_date.sh 0403-0957
# Example: ./scripts/export_setfit_onnx_by_date.sh 0407-1336

if [ -z "$1" ]; then
    echo "Usage: $0 <DATE_TIME>"
    echo "  DATE_TIME  : date-time like 0403-0957 or 0407-1336"
    exit 1
fi

DATE_TIME="$2"
DATE_TIME="$1"

BASE_DIRS=(
    "./trainer/checkpoints/setfit"
    "./trainer/checkpoints/sub"
)

RUNNER="python -m trainer.main sub setfit export-onnx"

SKIP_COUNT=0
EXPORT_COUNT=0
ERROR_COUNT=0

for BASE_DIR in "${BASE_DIRS[@]}"; do
    if [ ! -d "$BASE_DIR" ]; then
        continue
    fi

    for major_dir in "$BASE_DIR"/*/; do
        major_name=$(basename "$major_dir")

        # Try both setfit-{DATE_TIME} and supervised-{DATE_TIME} patterns
        for pattern in "setfit-$DATE_TIME" "supervised-$DATE_TIME"; do
            date_dir="${major_dir}${pattern}/"
            if [ ! -d "$date_dir" ]; then
                continue
            fi

            folder_name=$(basename "$date_dir")
            best_dir="${date_dir}best"
            onnx_path="${date_dir}best.onnx"

            if [ -d "$best_dir" ]; then
                if [ -f "$onnx_path" ]; then
                    echo "[WARN] [$major_name/$folder_name] ONNX already exists, skipping: $onnx_path"
                    SKIP_COUNT=$((SKIP_COUNT + 1))
                else
                    echo "[INFO] [$major_name/$folder_name] Exporting..."
                    echo "       from: $best_dir"
                    echo "       to  : $onnx_path"
                    $RUNNER -i "$best_dir" -o "$onnx_path"
                    if [ $? -eq 0 ]; then
                        EXPORT_COUNT=$((EXPORT_COUNT + 1))
                        echo "[OK]   [$major_name/$folder_name] Done"
                    else
                        ERROR_COUNT=$((ERROR_COUNT + 1))
                        echo "[FAIL] [$major_name/$folder_name] Export failed"
                    fi
                fi
            else
                echo "[WARN] [$major_name/$folder_name] No best/ subdir, skipping"
                SKIP_COUNT=$((SKIP_COUNT + 1))
            fi
            echo "---"
        done
    done
done

echo ""
echo "========== Done =========="
echo "Exported: $EXPORT_COUNT"
echo "Skipped:  $SKIP_COUNT"
echo "Errors:   $ERROR_COUNT"