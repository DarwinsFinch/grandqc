#!/bin/bash

# Script to process each SVS file individually with GrandQC
# Each SVS gets its own output subfolder

# Configuration
INPUT_DIR="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs"
BASE_OUTPUT_DIR="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs/QC_Results"
QC_MPP_MODEL=2.0
CREATE_GEOJSON="Y"

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

echo "========================================"
echo "GrandQC Processing - Individual SVS Files"
echo "Input folder: $INPUT_DIR"
echo "Output folder: $BASE_OUTPUT_DIR"
echo "MPP Model: $QC_MPP_MODEL (5x magnification)"
echo "========================================"
echo ""

# Counter for progress
total_files=$(ls "$INPUT_DIR"/*.svs 2>/dev/null | wc -l)
current=0

if [ "$total_files" -eq 0 ]; then
    echo "No SVS files found in $INPUT_DIR"
    exit 1
fi

# Process each SVS file
for svs_file in "$INPUT_DIR"/*.svs; do
    current=$((current + 1))
    
    # Get the base filename without extension
    basename=$(basename "$svs_file" .svs)
    
    # Create output subfolder for this SVS
    output_folder="$BASE_OUTPUT_DIR/$basename"
    mkdir -p "$output_folder"
    
    # Create temporary folder with just this one SVS file
    temp_folder="$BASE_OUTPUT_DIR/temp_single_svs"
    mkdir -p "$temp_folder"
    
    # Create symlink to the single SVS file
    ln -sf "$svs_file" "$temp_folder/$(basename "$svs_file")"
    
    echo "========================================"
    echo "[$current/$total_files] Processing: $basename"
    echo "Output: $output_folder"
    echo "========================================"
    
    # Step 1: Tissue Detection
    echo "Step 1/2: Running tissue detection..."
    python wsi_tis_detect.py \
        --slide_folder "$temp_folder" \
        --output_dir "$output_folder"
    
    if [ $? -ne 0 ]; then
        echo "✗ Error during tissue detection for $basename"
        rm -rf "$temp_folder"
        continue
    fi
    
    # Step 2: Artifact Segmentation
    echo "Step 2/2: Running artifact segmentation..."
    python main.py \
        --slide_folder "$temp_folder" \
        --output_dir "$output_folder" \
        --create_geojson "$CREATE_GEOJSON" \
        --mpp_model "$QC_MPP_MODEL"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $basename"
    else
        echo "✗ Error during artifact segmentation for $basename"
    fi
    
    # Clean up temp folder
    rm -rf "$temp_folder"
    
    echo ""
done

# Clean up any remaining temp folders
rm -rf "$BASE_OUTPUT_DIR/temp_single_svs"

echo "========================================"
echo "Processing completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "========================================"
