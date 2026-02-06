#!/bin/bash
# setting
SLIDE_FOLDER="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs"
OUTPUT_DIR="/home/lunas/Documents/Projects/GrandQC/data/ArtefactHeavyWSIs/Output"
QC_MPP_MODEL=2.0
CREATE_GEOJSON="Y"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

python main.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR" --create_geojson "$CREATE_GEOJSON" --mpp_model "$QC_MPP_MODEL"

echo "All processes completed!"
