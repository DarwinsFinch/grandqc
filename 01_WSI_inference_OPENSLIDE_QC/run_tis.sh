#!/bin/bash
# setting
SLIDE_FOLDER="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs"
OUTPUT_DIR="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs/Output"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "Tissue Segmentation is completed!"
