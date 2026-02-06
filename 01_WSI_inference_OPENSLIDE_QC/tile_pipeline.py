"""
tile_pipeline.py - GrandQC Tile-Based Pipeline

Performs tissue detection and artifact segmentation from pre-tiled images
(no WSI / OpenSlide needed). Produces the same outputs as run_tis.sh + run_art.sh.

Input tile format: VIPS Google layout  tiles_dir/0/{row}/{col}.jpg  (each 512x512)

Usage:
    python tile_pipeline.py \
        --tiles_1x /path/to/tiles_1x \
        --tiles_5x /path/to/tiles_5x \
        --output_dir /path/to/output \
        --slide_name SLIDE_001 \
        --mpp_model 2.0 \
        --create_geojson Y \
        --metadata_json /path/to/metadata.json
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import cv2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import timeit

# Import from existing GrandQC codebase
from wsi_colors import colors_QC7 as colors_art
from wsi_process import make_1class_map_thr, mask_to_geojson

Image.MAX_IMAGE_PIXELS = 1000000000

# ========================================================================
# CONFIGURATION
# ========================================================================

DEVICE = 'cuda'
"""
'cuda'   - NVIDIA GPU
'mps'    - Apple Silicon
'cpu'    - CPU fallback
"""

# Tissue Detection Model
MODEL_TD_DIR = './models/td/'
MODEL_TD_NAME = 'Tissue_Detection_MPP10.pth'
ENCODER_TD = 'timm-efficientnet-b0'
ENCODER_TD_WEIGHTS = 'imagenet'

# Artifact Model Directory
MODEL_QC_DIR = './models/qc/'
ENCODER_QC = 'timm-efficientnet-b0'
ENCODER_QC_WEIGHTS = 'imagenet'

# Tile / Patch size
TILE_SIZE = 512

# Background class for artifact detection
BACK_CLASS = 7

# Tissue detection colors
COLORS_TIS = [[50, 50, 250], [128, 128, 128]]  # BLUE: tissue, GRAY: background

# Overlay transparency
OVER_IMAGE = 0.7
OVER_MASK = 0.3


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def get_tile_grid(tile_dir):
    """
    Determine tile grid dimensions from VIPS Google layout.
    VIPS Google layout: tile_dir/0/{row}/{col}.jpg
    where row = y-index (vertical, top to bottom)
          col = x-index (horizontal, left to right)

    Returns: (n_rows, n_cols)
    """
    level_dir = os.path.join(tile_dir, '0')
    if not os.path.isdir(level_dir):
        raise FileNotFoundError(f"Expected directory {level_dir} not found. "
                                f"Tile dir should contain a '0/' subdirectory.")

    # Outer directories = rows (y indices)
    row_dirs = sorted(
        [d for d in os.listdir(level_dir)
         if os.path.isdir(os.path.join(level_dir, d))],
        key=int
    )
    n_rows = len(row_dirs)

    # Files in first row = columns (x indices)
    first_row_dir = os.path.join(level_dir, row_dirs[0])
    col_files = [f for f in os.listdir(first_row_dir) if f.endswith('.jpg')]
    n_cols = len(col_files)

    return n_rows, n_cols


def load_tile(tile_dir, row, col):
    """Load a single tile from VIPS Google layout: tile_dir/0/{row}/{col}.jpg"""
    path = os.path.join(tile_dir, '0', str(row), f'{col}.jpg')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tile not found: {path}")
    return Image.open(path).convert('RGB')


def to_tensor_x(x):
    """Convert HWC numpy array to CHW float32."""
    return x.transpose(2, 0, 1).astype('float32')


def make_class_map(mask, class_colors):
    """Create colored RGB class map from integer mask.
    Same logic as wsi_tis_detect_helper_fx.make_class_map."""
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, len(class_colors)):
        idx = mask == l
        r[idx] = class_colors[l][0]
        g[idx] = class_colors[l][1]
        b[idx] = class_colors[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def reconstruct_image_from_tiles(tile_dir, n_rows, n_cols, tile_size=512):
    """Reconstruct full image by stitching all tiles."""
    full_h = n_rows * tile_size
    full_w = n_cols * tile_size
    full_image = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    for row in range(n_rows):
        for col in range(n_cols):
            tile = load_tile(tile_dir, row, col)
            tile_arr = np.array(tile)[:tile_size, :tile_size, :3]
            full_image[row * tile_size:(row + 1) * tile_size,
                       col * tile_size:(col + 1) * tile_size] = tile_arr

    return full_image


# ========================================================================
# STEP 1: TISSUE DETECTION
# ========================================================================

def run_tissue_detection(tiles_1x_dir, output_dir, slide_name,
                         native_mpp=None, w_l0=None, h_l0=None):
    """
    Run tissue detection on 1x tiles (MPP ~10).

    Stitches all 1x tiles into a full thumbnail first, crops to exact
    dimensions (removing VIPS edge padding), then applies a single JPEG Q80
    compression to the whole image, then tiles and processes with the same
    overhang logic as wsi_tis_detect.py. This matches the original WSI
    pipeline's compression and tiling behavior.

    Produces (tile_ prefix to avoid overwriting WSI pipeline outputs):
        - tile_tis_det_mask/{slide_name}_MASK.png          (binary: 0=tissue, 1=background)
        - tile_tis_det_mask_col/{slide_name}_MASK_COL.png  (colored visualization)
        - tile_tis_det_overlay/{slide_name}_OVERLAY.jpg     (overlay on thumbnail)
        - tile_tis_det_thumbnail/{slide_name}.jpg           (reconstructed thumbnail)

    Returns: tissue mask as numpy array
    """
    print("\n" + "=" * 60)
    print("STEP 1: Tissue Detection")
    print("=" * 60)

    n_rows, n_cols = get_tile_grid(tiles_1x_dir)
    print(f"  Tile grid: {n_cols} cols x {n_rows} rows = {n_cols * n_rows} tiles")

    # Create output directories (tile_ prefix to not overwrite WSI outputs)
    dirs = {
        'mask': os.path.join(output_dir, 'tile_tis_det_mask'),
        'mask_col': os.path.join(output_dir, 'tile_tis_det_mask_col'),
        'overlay': os.path.join(output_dir, 'tile_tis_det_overlay'),
        'thumbnail': os.path.join(output_dir, 'tile_tis_det_thumbnail'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load tissue detection model
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_TD, ENCODER_TD_WEIGHTS)

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_TD,
        encoder_weights=ENCODER_TD_WEIGHTS,
        classes=2,
        activation=None,
    )
    model.load_state_dict(
        torch.load(os.path.join(MODEL_TD_DIR, MODEL_TD_NAME), map_location='cpu')
    )
    model.to(DEVICE)
    model.eval()

    # Step A: Reconstruct full thumbnail from tiles
    thumbnail = reconstruct_image_from_tiles(tiles_1x_dir, n_rows, n_cols, TILE_SIZE)

    # Crop to exact dimensions (remove VIPS edge padding)
    # This matches OpenSlide's get_thumbnail which returns exact image size
    if w_l0 is not None and h_l0 is not None and native_mpp is not None:
        MPP_MODEL_TD = 10.0
        reduction_factor = MPP_MODEL_TD / native_mpp
        exact_thumb_w = int(w_l0 // reduction_factor)
        exact_thumb_h = int(h_l0 // reduction_factor)
        # Clip to available grid in case metadata implies larger
        exact_thumb_w = min(exact_thumb_w, n_cols * TILE_SIZE)
        exact_thumb_h = min(exact_thumb_h, n_rows * TILE_SIZE)
        print(f"  Cropping thumbnail: ({n_cols*TILE_SIZE}, {n_rows*TILE_SIZE}) -> "
              f"({exact_thumb_w}, {exact_thumb_h})")
        thumbnail = thumbnail[:exact_thumb_h, :exact_thumb_w]

    thumbnail_img = Image.fromarray(thumbnail)

    # Save thumbnail (before JPEG compression, matching original)
    thumbnail_img.save(
        os.path.join(dirs['thumbnail'], slide_name + '.jpg'), quality=80)

    # Step B: JPEG compress the WHOLE image at Q80
    # (Matching wsi_tis_detect.py lines 93-97 exactly)
    image_arr = np.array(thumbnail_img)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, encoded = cv2.imencode('.jpg', image_arr, encode_param)
    image_arr = cv2.imdecode(encoded, 1)
    image = Image.fromarray(image_arr)

    width, height = image.size

    # Step C: Tile and process with overhang handling
    # (Matching wsi_tis_detect.py lines 101-152 exactly)
    p_s = TILE_SIZE
    wi_n = width // p_s
    he_n = height // p_s
    overhang_wi = width - wi_n * p_s
    overhang_he = height - he_n * p_s

    print(f"  Thumbnail: {width}x{height}, "
          f"Overhang: {overhang_wi}, {overhang_he}")

    for h in tqdm(range(he_n + 1), desc="  Tissue Detection"):
        for w in range(wi_n + 1):
            # Crop patch with overhang handling (same as original)
            if w != wi_n and h != he_n:
                image_work = image.crop(
                    (w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s))
            elif w == wi_n and h != he_n:
                image_work = image.crop(
                    (width - p_s, h * p_s, width, (h + 1) * p_s))
            elif w != wi_n and h == he_n:
                image_work = image.crop(
                    (w * p_s, height - p_s, (w + 1) * p_s, height))
            else:
                image_work = image.crop(
                    (width - p_s, height - p_s, width, height))

            # Preprocess and predict
            image_np = np.array(image_work)
            x = preprocessing_fn(image_np)
            x = to_tensor_x(x)
            x_tensor = torch.from_numpy(x).to(DEVICE).unsqueeze(0)

            predictions = model.predict(x_tensor)
            predictions = predictions.squeeze().cpu().numpy()

            mask = np.argmax(predictions, axis=0).astype('int8')
            class_mask = make_class_map(mask, COLORS_TIS)

            # Stitch with overhang handling (same as original)
            if w == 0:
                temp_mask = mask
                temp_class = class_mask
            elif w == wi_n:
                mask = mask[:, p_s - overhang_wi:p_s]
                temp_mask = np.concatenate((temp_mask, mask), axis=1)
                class_mask = class_mask[:, p_s - overhang_wi:p_s, :]
                temp_class = np.concatenate((temp_class, class_mask), axis=1)
            else:
                temp_mask = np.concatenate((temp_mask, mask), axis=1)
                temp_class = np.concatenate((temp_class, class_mask), axis=1)

        if h == 0:
            full_mask = temp_mask
            full_class_map = temp_class
        elif h == he_n:
            temp_mask = temp_mask[p_s - overhang_he:p_s,]
            full_mask = np.concatenate((full_mask, temp_mask), axis=0)
            temp_class = temp_class[p_s - overhang_he:p_s, :, :]
            full_class_map = np.concatenate((full_class_map, temp_class), axis=0)
        else:
            full_mask = np.concatenate((full_mask, temp_mask), axis=0)
            full_class_map = np.concatenate((full_class_map, temp_class), axis=0)

    # Save tissue mask (binary)
    Image.fromarray(full_mask.astype(np.uint8)).save(
        os.path.join(dirs['mask'], slide_name + '_MASK.png'))

    # Save colored mask
    Image.fromarray(full_class_map).save(
        os.path.join(dirs['mask_col'], slide_name + '_MASK_COL.png'))

    # Create overlay (using JPEG-compressed image, matching original)
    overlay = cv2.addWeighted(
        np.array(image), OVER_IMAGE, full_class_map, OVER_MASK, 0)
    Image.fromarray(overlay).save(
        os.path.join(dirs['overlay'], slide_name + '_OVERLAY.jpg'))

    print(f"  Tissue detection completed")
    return full_mask


# ========================================================================
# QC SUMMARY HELPER
# ========================================================================

def _write_qc_summary(summary_path, slide_name, tissue_mask, artifact_mask,
                      exact_w, exact_h, active_w, active_h,
                      buffer_right, buffer_bottom,
                      native_mpp, obj_power, mpp_model, elapsed_min):
    """
    Write a human-readable QC summary with tissue and artifact percentages.

    Computes:
        - % of image that is tissue (from tissue detection mask)
        - % of tissue that is artifact (classes 2-6 in artifact mask)
        - Per-artifact-class % of tissue pixels
    """
    # --- Class definitions ---
    ARTIFACT_CLASSES = {
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "Out of Focus (OOF)",
    }

    # --- Tissue percentage (from 1x tissue mask) ---
    # tissue_mask: 0 = tissue, 1 = background
    total_tissue_mask_px = tissue_mask.size
    tissue_px = int(np.count_nonzero(tissue_mask == 0))
    bg_px = int(np.count_nonzero(tissue_mask == 1))
    tissue_pct = 100.0 * tissue_px / total_tissue_mask_px if total_tissue_mask_px > 0 else 0.0

    # --- Artifact percentages (from 5x artifact mask, active region only) ---
    # Active region = the area actually processed by the model (excludes buffer)
    active_mask = artifact_mask[:active_h, :active_w]
    # In the artifact mask within the active region:
    #   Class 1 = Normal Tissue (clean)
    #   Classes 2-6 = Artifacts
    #   Class 7 = Background (non-tissue, skipped by model)

    # Tissue in artifact mask = everything that is NOT background (cls 7)
    # and NOT buffer (cls 0), i.e. classes 1-6
    tissue_in_art = int(np.count_nonzero((active_mask >= 1) & (active_mask <= 6)))
    artifact_total = 0
    class_counts = {}
    for cls_id in ARTIFACT_CLASSES:
        count = int(np.count_nonzero(active_mask == cls_id))
        class_counts[cls_id] = count
        artifact_total += count
    clean_tissue = int(np.count_nonzero(active_mask == 1))

    artifact_pct_of_tissue = (100.0 * artifact_total / tissue_in_art
                              if tissue_in_art > 0 else 0.0)
    clean_pct_of_tissue = (100.0 * clean_tissue / tissue_in_art
                           if tissue_in_art > 0 else 0.0)

    # --- Write summary ---
    with open(summary_path, 'w') as f:
        f.write(f"{'='*65}\n")
        f.write(f"GrandQC - Quality Control Summary\n")
        f.write(f"{'='*65}\n\n")
        f.write(f"Slide:           {slide_name}\n")
        f.write(f"Objective power: {obj_power}x\n" if obj_power else "")
        f.write(f"Native MPP:      {native_mpp}\n" if native_mpp else "")
        f.write(f"Model MPP:       {mpp_model}\n")
        f.write(f"Processing time: {elapsed_min} min\n")
        f.write(f"\n{'='*65}\n")
        f.write(f"TISSUE DETECTION (at 1x / MPP 10)\n")
        f.write(f"{'='*65}\n")
        f.write(f"  Total image pixels:      {total_tissue_mask_px:>12,}\n")
        f.write(f"  Tissue pixels:           {tissue_px:>12,}  "
                f"({tissue_pct:.2f}%)\n")
        f.write(f"  Background pixels:       {bg_px:>12,}  "
                f"({100.0 - tissue_pct:.2f}%)\n")
        f.write(f"\n{'='*65}\n")
        f.write(f"ARTIFACT DETECTION (at {mpp_model} MPP)\n")
        f.write(f"{'='*65}\n")
        f.write(f"  Active region:           {active_w}x{active_h} px "
                f"({active_w * active_h:,} px)\n")
        f.write(f"  Tissue pixels (cls 1-6): {tissue_in_art:>12,}\n")
        f.write(f"  Clean tissue (cls 1):    {clean_tissue:>12,}  "
                f"({clean_pct_of_tissue:.2f}% of tissue)\n")
        f.write(f"  Artifact pixels (cls 2-6): "
                f"{artifact_total:>10,}  "
                f"({artifact_pct_of_tissue:.2f}% of tissue)\n")
        f.write(f"\n  {'Artifact Class':<30} {'Pixels':>12} "
                f"{'% of Tissue':>12} {'% of Artifacts':>14}\n")
        f.write(f"  {'-'*68}\n")
        for cls_id, cls_name in ARTIFACT_CLASSES.items():
            count = class_counts[cls_id]
            pct_tissue = (100.0 * count / tissue_in_art
                          if tissue_in_art > 0 else 0.0)
            pct_art = (100.0 * count / artifact_total
                       if artifact_total > 0 else 0.0)
            f.write(f"  {cls_name:<30} {count:>12,} "
                    f"{pct_tissue:>11.2f}% {pct_art:>13.2f}%\n")
        f.write(f"\n{'='*65}\n")

    print(f"  QC summary saved: {summary_path}")


# ========================================================================
# STEP 2: ARTIFACT DETECTION
# ========================================================================

def run_artifact_detection(tiles_5x_dir, tiles_1x_dir, tissue_mask,
                           output_dir, slide_name,
                           mpp_model=2.0, create_geojson="Y",
                           native_mpp=None, obj_power=None,
                           w_l0=None, h_l0=None):
    """
    Run artifact detection on 5x tiles (MPP ~2.0) using tissue mask.

    Produces (tile_ prefix to avoid overwriting WSI pipeline outputs):
        - tile_maps_qc/{slide_name}_map_QC.png          (colored artifact map)
        - tile_mask_qc/{slide_name}_mask.png             (raw class mask)
        - tile_overlays_qc/{slide_name}_overlay_QC.jpg   (overlay on thumbnail)
        - tile_geojson_qc/{slide_name}.geojson           (GeoJSON, optional)
        - tile_report_*_stats_per_slide.txt              (statistics report)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Artifact Detection")
    print("=" * 60)

    n_rows_5x, n_cols_5x = get_tile_grid(tiles_5x_dir)
    n_rows_1x, n_cols_1x = get_tile_grid(tiles_1x_dir)

    print(f"  5x tile grid: {n_cols_5x} cols x {n_rows_5x} rows = {n_cols_5x * n_rows_5x} tiles")

    # Select artifact model checkpoint
    if mpp_model == 1.5:
        model_qc_name = 'GrandQC_MPP15.pth'
    elif mpp_model == 1.0:
        model_qc_name = 'GrandQC_MPP1.pth'
    elif mpp_model == 2.0:
        model_qc_name = 'GrandQC_MPP2.pth'
    else:
        raise ValueError("mpp_model must be 1.0, 1.5, or 2.0")

    # Create output directories (tile_ prefix to not overwrite WSI outputs)
    maps_dir = os.path.join(output_dir, 'tile_maps_qc')
    overlay_dir = os.path.join(output_dir, 'tile_overlays_qc')
    mask_dir = os.path.join(output_dir, 'tile_mask_qc')
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if create_geojson == "Y":
        geojson_dir = os.path.join(output_dir, 'tile_geojson_qc')
        os.makedirs(geojson_dir, exist_ok=True)

    # Load artifact model (saved as full model object, not state_dict)
    model = torch.load(
        os.path.join(MODEL_QC_DIR, model_qc_name), map_location=DEVICE)

    # Preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_QC, ENCODER_QC_WEIGHTS)
    model_size = (TILE_SIZE, TILE_SIZE)

    # ---- Compute WSI-equivalent patch counts and buffer sizes ----
    # This matches slide_info() + slide_process_single() from the WSI pipeline
    tis_mask_img = Image.fromarray(tissue_mask.astype(np.uint8))
    grid_w = n_cols_5x * TILE_SIZE
    grid_h = n_rows_5x * TILE_SIZE

    if w_l0 is not None and h_l0 is not None and native_mpp is not None:
        # Compute WSI-equivalent patch geometry
        p_s_l0 = int(mpp_model / native_mpp * TILE_SIZE)  # level 0 patch size
        patch_n_w = int(w_l0 / p_s_l0)  # number of patches (width)
        patch_n_h = int(h_l0 / p_s_l0)  # number of patches (height)
        # Buffer at edges (matching WSI slide_process_single)
        buffer_right = int((w_l0 - patch_n_w * p_s_l0) * native_mpp / mpp_model)
        buffer_bottom = int((h_l0 - patch_n_h * p_s_l0) * native_mpp / mpp_model)
        # Exact output dimensions (matching WSI mask size)
        exact_w = patch_n_w * TILE_SIZE + buffer_right
        exact_h = patch_n_h * TILE_SIZE + buffer_bottom
        active_w = patch_n_w * TILE_SIZE
        active_h = patch_n_h * TILE_SIZE

        print(f"  WSI-equivalent patches: {patch_n_w}x{patch_n_h} "
              f"(p_s_l0={p_s_l0})")
        print(f"  Buffer: right={buffer_right}, bottom={buffer_bottom}")
        print(f"  Exact output: {exact_w}x{exact_h} "
              f"(active: {active_w}x{active_h})")

        # Resize tissue mask to exact dimensions (matching main.py line 143)
        tis_det_map_resized = np.array(
            tis_mask_img.resize((exact_w, exact_h), Image.Resampling.LANCZOS))
    else:
        # Fallback: use tile grid dimensions
        print("  Warning: w_l0/h_l0/native_mpp not available, "
              "using tile grid dimensions for tissue mask resize")
        tis_det_map_resized = np.array(
            tis_mask_img.resize((grid_w, grid_h), Image.Resampling.LANCZOS))
        patch_n_w = n_cols_5x
        patch_n_h = n_rows_5x
        exact_w = grid_w
        exact_h = grid_h
        active_w = grid_w
        active_h = grid_h
        buffer_right = 0
        buffer_bottom = 0
        p_s_l0 = int(mpp_model / 0.25 * TILE_SIZE) if native_mpp is None else int(mpp_model / native_mpp * TILE_SIZE)

    print(f"  Tissue mask resized: {tissue_mask.shape} -> {tis_det_map_resized.shape}")

    # Pre-allocate output mask at EXACT dimensions (filled with 0, matching WSI buffer)
    # WSI pipeline: active region has model predictions, buffer region is 0
    full_mask = np.zeros((exact_h, exact_w), dtype=np.uint8)

    start_time = timeit.default_timer()

    # Process only tiles in the ACTIVE region (matching WSI patch_n_w x patch_n_h)
    # Edge tiles beyond this are part of the buffer (filled with 0, matching WSI)
    tissue_tiles = 0
    skipped_tiles = 0

    for row in tqdm(range(patch_n_h), desc="  Artifact Detection"):
        for col in range(patch_n_w):
            # Get corresponding tissue mask patch
            r_s, r_e = row * TILE_SIZE, (row + 1) * TILE_SIZE
            c_s, c_e = col * TILE_SIZE, (col + 1) * TILE_SIZE
            td_patch = tis_det_map_resized[r_s:r_e, c_s:c_e]

            # Handle edge patches that might be smaller than TILE_SIZE
            if td_patch.shape != (TILE_SIZE, TILE_SIZE):
                padding = [(0, TILE_SIZE - td_patch.shape[0]),
                           (0, TILE_SIZE - td_patch.shape[1])]
                td_patch_padded = np.pad(td_patch, padding,
                                         mode='constant', constant_values=1)
            else:
                td_patch_padded = td_patch

            # Check if enough tissue pixels (>50 pixels of class 0 = tissue)
            if np.count_nonzero(td_patch == 0) > 50:
                tissue_tiles += 1

                # Load tile from 5x grid
                # Tile indices match directly since tiles are contiguous
                if col < n_cols_5x and row < n_rows_5x:
                    tile_img = load_tile(tiles_5x_dir, row, col)
                else:
                    # Beyond tile grid - shouldn't happen but handle gracefully
                    skipped_tiles += 1
                    full_mask[r_s:r_e, c_s:c_e] = BACK_CLASS
                    continue

                # Ensure correct size (should already be 512x512)
                if tile_img.size != model_size:
                    tile_img = tile_img.resize(model_size, Image.Resampling.LANCZOS)
                    print('  Warning: tile resized to 512x512')

                # Preprocess (same as wsi_process.get_preprocessing)
                image_np = np.array(tile_img)
                x = preprocessing_fn(image_np)
                x = to_tensor_x(x)
                x_tensor = torch.from_numpy(x).to(DEVICE).unsqueeze(0)

                # Predict
                predictions = model.predict(x_tensor)
                predictions = predictions.squeeze().cpu().numpy()

                mask_raw = np.argmax(predictions, axis=0).astype('int8')

                # Override background regions using tissue mask
                mask = np.where(td_patch_padded == 1, BACK_CLASS, mask_raw)
            else:
                skipped_tiles += 1
                mask = np.full((TILE_SIZE, TILE_SIZE), BACK_CLASS, dtype=np.int8)

            # Place in output mask (active region only)
            full_mask[r_s:r_e, c_s:c_e] = mask

    stop_time = timeit.default_timer()
    elapsed_min = round((stop_time - start_time) / 60, 1)

    print(f"  Processed {tissue_tiles} tissue tiles, "
          f"skipped {skipped_tiles} background tiles")
    print(f"  Time: {elapsed_min} min")

    # ---- Save colored artifact map ----
    colored_map = make_1class_map_thr(full_mask, colors_art)
    colored_map_img = Image.fromarray(colored_map)
    # Resize for display (same as original: patch_n_w * 50 x patch_n_h * 50)
    colored_map_img = colored_map_img.resize(
        (patch_n_w * 50, patch_n_h * 50), Image.Resampling.LANCZOS)
    map_path = os.path.join(maps_dir, slide_name + "_map_QC.png")
    colored_map_img.save(map_path)

    # ---- Save raw mask ----
    mask_path = os.path.join(mask_dir, slide_name + "_mask.png")
    cv2.imwrite(mask_path, full_mask)

    # ---- GeoJSON ----
    if create_geojson == "Y":
        geojson_path = os.path.join(geojson_dir, slide_name + '.geojson')
        if native_mpp is not None:
            factor = mpp_model / native_mpp
        else:
            factor = 1.0
            print("  Warning: native_mpp not provided, "
                  "GeoJSON coordinates in model pixels")
        mask_to_geojson(mask_path, geojson_path, factor)

    # ---- Create overlay ----
    # Reconstruct thumbnail from 1x tiles, cropped to exact dimensions
    thumbnail = reconstruct_image_from_tiles(
        tiles_1x_dir, n_rows_1x, n_cols_1x, TILE_SIZE)
    if w_l0 is not None and h_l0 is not None and native_mpp is not None:
        MPP_MODEL_TD = 10.0
        reduction_factor = MPP_MODEL_TD / native_mpp
        crop_w = min(int(w_l0 // reduction_factor), n_cols_1x * TILE_SIZE)
        crop_h = min(int(h_l0 // reduction_factor), n_rows_1x * TILE_SIZE)
        thumbnail = thumbnail[:crop_h, :crop_w]
    thumbnail_img = Image.fromarray(thumbnail)
    heatmap_resized = colored_map_img.resize(
        thumbnail_img.size, Image.Resampling.LANCZOS)
    overlay = cv2.addWeighted(
        np.array(thumbnail_img), OVER_IMAGE,
        np.array(heatmap_resized), OVER_MASK, 0)
    overlay_path = os.path.join(overlay_dir, slide_name + "_overlay_QC.jpg")
    Image.fromarray(overlay).save(overlay_path)

    # ---- Write report ----
    case_name = os.path.basename(output_dir)
    report_name = f'tile_report_{case_name}_0_1_stats_per_slide.txt'
    report_path = os.path.join(output_dir, report_name)

    mpp_str = str(native_mpp) if native_mpp else "N/A"
    obj_str = str(obj_power) if obj_power else "N/A"

    with open(report_path, 'w') as f:
        f.write("slide_name\tobj_power\tmpp\t"
                "patch_n_h\tpatch_n_w\tpatch_overall\t"
                "height\twidth\ttime_min\n")
        f.write(f"{slide_name}\t{obj_str}\t{mpp_str}\t"
                f"{patch_n_h}\t{patch_n_w}\t{patch_n_h * patch_n_w}\t"
                f"{patch_n_h * p_s_l0}\t{patch_n_w * p_s_l0}\t"
                f"{elapsed_min}\n")

    # ---- Write QC summary (tissue & artifact statistics) ----
    summary_path = os.path.join(output_dir,
                                f'tile_qc_summary_{slide_name}.txt')
    _write_qc_summary(summary_path, slide_name, tissue_mask, full_mask,
                      exact_w, exact_h, active_w, active_h,
                      buffer_right, buffer_bottom,
                      native_mpp, obj_power, mpp_model, elapsed_min)

    print(f"  Artifact detection completed")

    # Clean up
    del full_mask, colored_map


# ========================================================================
# MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GrandQC Tile-Based Pipeline: '
                    'Tissue + Artifact Detection from pre-tiled images')

    parser.add_argument('--tiles_1x', required=True,
                        help='Path to 1x tile directory '
                             '(contains 0/{row}/{col}.jpg)')
    parser.add_argument('--tiles_5x', required=True,
                        help='Path to 5x tile directory '
                             '(contains 0/{row}/{col}.jpg)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--slide_name', required=True,
                        help='Name for output files')
    parser.add_argument('--mpp_model', default=2.0, type=float,
                        help='MPP of artifact model: 1.0, 1.5, or 2.0 '
                             '(default: 2.0)')
    parser.add_argument('--create_geojson', default='Y', type=str,
                        help='Create GeoJSON output: Y or N (default: Y)')
    parser.add_argument('--native_mpp', default=None, type=float,
                        help='Native MPP of original slide '
                             '(for GeoJSON coordinate scaling)')
    parser.add_argument('--obj_power', default=None, type=int,
                        help='Objective power of original slide '
                             '(for report, optional)')
    parser.add_argument('--metadata_json', default=None, type=str,
                        help='Path to metadata.json with native_mpp, '
                             'obj_power (alternative to individual args)')

    args = parser.parse_args()

    # Load metadata from JSON if provided
    native_mpp = args.native_mpp
    obj_power = args.obj_power
    w_l0 = None
    h_l0 = None

    if args.metadata_json and os.path.exists(args.metadata_json):
        with open(args.metadata_json) as f:
            meta = json.load(f)
        if native_mpp is None:
            native_mpp = meta.get('native_mpp')
        if obj_power is None:
            obj_power = meta.get('obj_power')
        w_l0 = meta.get('w_l0')
        h_l0 = meta.get('h_l0')
        print(f"Loaded metadata: MPP={native_mpp}, Obj Power={obj_power}x, "
              f"Dims={w_l0}x{h_l0}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GrandQC Tile Pipeline")
    print("=" * 60)
    print(f"  Slide:      {args.slide_name}")
    print(f"  1x tiles:   {args.tiles_1x}")
    print(f"  5x tiles:   {args.tiles_5x}")
    print(f"  Output:     {args.output_dir}")
    print(f"  MPP Model:  {args.mpp_model}")
    print(f"  Native MPP: {native_mpp}")
    print(f"  GeoJSON:    {args.create_geojson}")

    # ---- Step 1: Tissue Detection ----
    tissue_mask = run_tissue_detection(
        args.tiles_1x, args.output_dir, args.slide_name,
        native_mpp=native_mpp, w_l0=w_l0, h_l0=h_l0
    )

    # ---- Step 2: Artifact Detection ----
    run_artifact_detection(
        args.tiles_5x, args.tiles_1x, tissue_mask,
        args.output_dir, args.slide_name,
        mpp_model=args.mpp_model,
        create_geojson=args.create_geojson,
        native_mpp=native_mpp,
        obj_power=obj_power,
        w_l0=w_l0,
        h_l0=h_l0
    )

    print("\n" + "=" * 60)
    print(f"Pipeline completed for {args.slide_name}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
