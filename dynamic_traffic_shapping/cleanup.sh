#!/bin/bash

echo "Cleaning up unnecessary files in dynamic_traffic_shapping directory..."

# Files that are essential and should NOT be deleted
ESSENTIAL_FILES=(
    "video_streamer.py"
    "receive_video.py"
    "dynamic_tc_control.sh"
    "run_smooth_video.py"
    "README.md"
    "DETAILED_CODE_DOCUMENTATION.md"
    "cleanup.sh"
)

# Count how many files will be deleted
TO_DELETE=0
for file in *; do
    if [[ ! " ${ESSENTIAL_FILES[@]} " =~ " ${file} " ]] && [[ -f "$file" ]]; then
        ((TO_DELETE++))
    fi
done

echo "Found $TO_DELETE files that can be safely deleted."
echo "The following files will be kept:"
for file in "${ESSENTIAL_FILES[@]}"; do
    echo "  - $file"
done

echo ""
echo "Deleting unnecessary files..."

for file in *; do
    if [[ ! " ${ESSENTIAL_FILES[@]} " =~ " ${file} " ]] && [[ -f "$file" ]]; then
        echo "Deleting: $file"
        rm "$file"
    fi
done

# Also remove the test_results directory if it exists
if [[ -d "test_results" ]]; then
    echo "Deleting: test_results/ directory"
    rm -rf "test_results"
fi

echo "Cleanup complete!"