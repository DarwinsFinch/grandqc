#!/bin/bash
# Run GrandQC tile-based pipeline for all tile sets in GrandQC_Tiles/
# Each subfolder should contain: tiles_1x/, tiles_5x/, metadata.json

TILES_BASE="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs/GrandQC_Tiles"
OUTPUT_BASE="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs/GrandQC_Results"
QC_MPP_MODEL=2.0
CREATE_GEOJSON="Y"

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "GrandQC Tile Pipeline"
echo "Tiles:  $TILES_BASE"
echo "Output: $OUTPUT_BASE"
echo "=========================================="

for tile_dir in "$TILES_BASE"/*/; do
    slide_name=$(basename "$tile_dir")
    output_dir="$OUTPUT_BASE/$slide_name"

    tiles_1x="$tile_dir/tiles_1x"
    tiles_5x="$tile_dir/tiles_5x"
    metadata="$tile_dir/metadata.json"

    # Check required directories exist
    if [ ! -d "$tiles_1x" ] || [ ! -d "$tiles_5x" ]; then
        echo "Skipping $slide_name: missing tiles_1x or tiles_5x"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Processing: $slide_name"
    echo "=========================================="

    python tile_pipeline.py \
        --tiles_1x "$tiles_1x" \
        --tiles_5x "$tiles_5x" \
        --output_dir "$output_dir" \
        --slide_name "$slide_name" \
        --mpp_model "$QC_MPP_MODEL" \
        --create_geojson "$CREATE_GEOJSON" \
        --metadata_json "$metadata"

    if [ $? -eq 0 ]; then
        echo "Done: $slide_name"
    else
        echo "ERROR processing $slide_name"
    fi
done

echo ""
echo "=========================================="
echo "All slides processed!"
echo "Results in: $OUTPUT_BASE"
echo "=========================================="
