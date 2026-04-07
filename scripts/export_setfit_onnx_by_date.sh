#!/bin/bash
# Export SetFit ONNX models by date folder.
# For each major category subdir under BASE_DIR, finds folders matching setfit-{DATE_TIME}
# and exports ONNX from best/ subfolder to the parent folder (best.onnx).
#
# Usage: ./scripts/export_setfit_onnx_by_date.sh <BASE_DIR> <DATE_TIME>
# Example: ./scripts/export_setfit_onnx_by_date.sh ./trainer/checkpoints/setfit 0403-0957

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <BASE_DIR> <DATE_TIME>"
    echo "  BASE_DIR   : e.g. ./trainer/checkpoints/setfit"
    echo "  DATE_TIME  : date-time like 0403-0957"
    echo ""
    echo "Example: $0 ./trainer/checkpoints/setfit 0403-0957"
    exit 1
fi

BASE_DIR="$1"
DATE_TIME="$2"

if [ ! -d "$BASE_DIR" ]; then
    echo "[ERROR] BASE_DIR not found: $BASE_DIR"
    exit 1
fi

RUNNER="python -m trainer.main sub setfit export-onnx"

SKIP_COUNT=0
EXPORT_COUNT=0
ERROR_COUNT=0

for major_dir in "$BASE_DIR"/*/; do
    major_name=$(basename "$major_dir")

    # Find setfit-{DATE_TIME} folder (exact match)
    for date_dir in "$major_dir"setfit-"$DATE_TIME"/; do
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

echo ""
echo "========== Done =========="
echo "Exported: $EXPORT_COUNT"
echo "Skipped:  $SKIP_COUNT"
echo "Errors:   $ERROR_COUNT"