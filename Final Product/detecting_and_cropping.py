from pathlib import Path
import random
import numpy as np
import cv2
import pandas as pd
"""
detecting_and_cropping.py

Usage:
  python detecting_and_cropping.py

Default:
  expects ../data (relative to this script file)
  writes   ../crops
"""
# --- Parameters --- In this section parameters could be edited as required
# All Parameters are None Safe, to remove a filter please set it is None
THRESH       = 180 # The mask threshold (lower = more insect morphs however it increases the change of non-insect crops)
MIN_WIDTH    = 50 # The min width filter of the cv2 bounding box size
MIN_HEIGHT   = 50 # The min height filter of the cv2 bounding box size
MAX_WIDTH    = 800 # The max width filter of the cv2 bounding box size
MAX_HEIGHT   = 800 # The max height filter of the cv2 bounding box size
MIN_AREA     = 1500 # The min area filter of the cv2 bounding box size
MAX_AREA     = 100000 # The max area filter of the cv2 bounding box size
EDGE_MARGIN  = 20 # The number of pixels as an edge margin to prevent lower lighting and resolution crops.
MAX_IMAGES   = None # The number of images to use from ../data, mainly used as a test run. None = all images.


SEED = 42

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Base paths relative to this script (so it works from any terminal folder)
    script_dir = Path(__file__).resolve().parent
    img_dir = (script_dir / "../data").resolve()
    save_root = (script_dir / "../crops").resolve()
    kept_dir  = save_root / "kept"
    vis_dir   = save_root / "visualizations"

    print("Data dir:", img_dir)
    print("Exists:", img_dir.exists())
    print("Files found:", len(list(img_dir.glob("*.*"))))

    for d in [save_root, kept_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(img_dir.glob("*.jpg")))
    if MAX_IMAGES is not None:
        image_files = image_files[:MAX_IMAGES]
    print(f"📂 Processing {len(image_files)} images from {img_dir}")

    log_records = []

    for img_idx, img_path in enumerate(image_files, 1):
        img_name = img_path.stem
        print(f"\n[{img_idx}/{len(image_files)}] Processing {img_name}...")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not read {img_path.name}")
            continue

        orig = img.copy()
        h_img, w_img = img.shape[:2]

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        high_contrast = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        inverted = cv2.bitwise_not(high_contrast)
        gray_inv = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray_inv, THRESH, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        saved, skipped = 0, 0

        vis_kept = orig.copy()
        vis_skipped = orig.copy()

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            crop = orig[y:y+h, x:x+w]

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            std = gray_crop.std()
            bright_fraction = np.mean(gray_crop > 200)
            dark_fraction = np.mean(gray_crop < 50)
            min_val, max_val, _, _ = cv2.minMaxLoc(gray_crop)
            reason = "kept"

            # --- Filtering rules (None-safe) ---
            if (
                (MIN_AREA is not None and area < MIN_AREA) or
                (MAX_AREA is not None and area > MAX_AREA)
            ):
                reason = "area"
            elif (
                (MIN_WIDTH  is not None and w < MIN_WIDTH)  or
                (MAX_WIDTH  is not None and w > MAX_WIDTH)  or
                (MIN_HEIGHT is not None and h < MIN_HEIGHT) or
                (MAX_HEIGHT is not None and h > MAX_HEIGHT)
            ):
                reason = "size"
            elif (
                EDGE_MARGIN is not None and (
                    x < EDGE_MARGIN or y < EDGE_MARGIN or
                    x + w > w_img - EDGE_MARGIN or
                    y + h > h_img - EDGE_MARGIN
                )
            ):
                reason = "edge"
            elif std < 15 or (bright_fraction < 0.001 and dark_fraction < 0.001):
                reason = "flat"

            info = f"{area:.0f}_{x}_{y}_{w}_{h}_{aspect_ratio:.2f}_{std:.2f}_{bright_fraction:.3f}_{dark_fraction:.3f}_{min_val:.0f}_{max_val:.0f}_{reason}"
            crop_name = f"{img_idx:03d}_{i+1:02d}_{info}.png"
            out_path = kept_dir / crop_name

            if reason == "kept":
                color = (0, 255, 0)
                cv2.rectangle(vis_kept, (x, y), (x + w, y + h), color, 3)
                cv2.putText(vis_kept, f"{i}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
                cv2.imwrite(str(out_path), crop)
                saved += 1
            else:
                color = (0, 0, 255)
                cv2.rectangle(vis_skipped, (x, y), (x + w, y + h), color, 3)
                cv2.putText(vis_skipped, f"{i}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
                skipped += 1

            log_records.append({
                "image_index": img_idx,
                "crop_index": i + 1,
                "image_name": img_name,
                "area": area,
                "x": x, "y": y, "w": w, "h": h,
                "aspect_ratio": aspect_ratio,
                "std": std,
                "bright_fraction": bright_fraction,
                "dark_fraction": dark_fraction,
                "min_val": min_val,
                "max_val": max_val,
                "reason": reason,
                "filename": crop_name if reason == "kept" else None,
                "abs_crop_path": str(out_path.resolve()) if reason == "kept" else None
            })

        vis_original = vis_dir / f"{img_idx:03d}_original.png"
        vis_kept_path = vis_dir / f"{img_idx:03d}_kept.png"
        vis_skipped_path = vis_dir / f"{img_idx:03d}_skipped.png"
        cv2.imwrite(str(vis_original), orig)
        cv2.imwrite(str(vis_kept_path), vis_kept)
        cv2.imwrite(str(vis_skipped_path), vis_skipped)

        log_records.append({
            "image_index": img_idx,
            "crop_index": None,
            "image_name": img_name,
            "reason": "visualization",
            "abs_original_path": str(vis_original),
            "abs_kept_vis_path": str(vis_kept_path),
            "abs_skipped_vis_path": str(vis_skipped_path)
        })

        print(f"✅ {saved} kept, 🚫 {skipped} skipped in {img_name}")

    csv_path = save_root / "crop_log.csv"
    pd.DataFrame(log_records).to_csv(csv_path, index=False)
    print("\n📄 CSV log saved to:", csv_path)
    print(f"✅ All done! Kept crops → {kept_dir}")

if __name__ == "__main__":
    main()
