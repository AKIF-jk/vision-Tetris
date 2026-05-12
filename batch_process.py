import glob
import os
import sys

import cv2
import numpy as np

import config
from main import run_pipeline


def stitch_images(img1, img2, img3):
    max_h = max(img1.shape[0], img2.shape[0], img3.shape[0])
    total_w = img1.shape[1] + img2.shape[1] + img3.shape[1]

    def pad(img):
        h, w = img.shape[:2]
        top = (max_h - h) // 2
        bottom = max_h - h - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.hstack([pad(img1), pad(img2), pad(img3)])


def add_labels(stitched, label1="Original", label2="Annotated", label3="Heatmap"):
    result = stitched.copy()
    h, w = result.shape[:2]
    panel_w = w // 3

    for i, label in enumerate([label1, label2, label3]):
        x = panel_w * i + 10
        cv2.putText(result, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def main():
    image_paths = sorted(glob.glob("data/**/*.jpg", recursive=True))
    if not image_paths:
        print("No images found in data/")
        sys.exit(1)

    os.makedirs("output/results", exist_ok=True)

    for input_path in image_paths:
        prefix = os.path.splitext(os.path.basename(input_path))[0]
        result_path = os.path.join("output/results", f"{prefix}_stitched.jpg")
        if os.path.exists(result_path):
            print(f"  SKIP  {prefix} (already exists)")
            continue

        print(f"  PROC  {prefix}")
        run_pipeline(input_path, debug=False, verbose=False)

        annotated_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_annotated.jpg")
        heatmap_path = os.path.join(config.HEATMAP_DIR, f"{prefix}_heatmap.jpg")

        orig = cv2.imread(input_path)
        ann = cv2.imread(annotated_path)
        hm = cv2.imread(heatmap_path)

        if orig is None:
            print(f"  FAIL  {prefix} — cannot read original")
            continue
        if ann is None:
            print(f"  FAIL  {prefix} — annotated not found at {annotated_path}")
            continue
        if hm is None:
            print(f"  FAIL  {prefix} — heatmap not found at {heatmap_path}")
            continue

        stitched = stitch_images(orig, ann, hm)
        stitched = add_labels(stitched)
        cv2.imwrite(result_path, stitched)
        print(f"  DONE  {prefix} -> {result_path}")

    print(f"\nAll done. Results in output/results/ ({len(image_paths)} images processed)")


if __name__ == "__main__":
    main()
