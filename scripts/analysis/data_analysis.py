"""Analyzing statistics of the TCGA-BRCA dataset."""

import os

import numpy as np
from openslide import OpenSlide

from augur.datasets.utils import load_slide_records, resolve_manifest_path


def get_mpp_statistics():
    """Extract and print statistics of microns-per-pixel (MPP) values from the TCGA-BRCA dataset."""
    root_dir = "data/TCGA-BRCA"
    manifest_path = resolve_manifest_path(root_dir, None)

    slide_records = load_slide_records(
        manifest_path=manifest_path,
        ordered_data_dir=os.path.join(root_dir, "ordered_data"),
        max_slides=None,
    )

    mppx_values = []
    mppy_values = []
    missing_ids = []

    for record in slide_records:
        slide = OpenSlide(record.slide_path)
        try:
            mpp_x = float(slide.properties.get("openslide.mpp-x"))
            mpp_y = float(slide.properties.get("openslide.mpp-y"))
            mppx_values.append(mpp_x)
            mppy_values.append(mpp_y)
        except (TypeError, ValueError):
            print("".center(80, "-"))
            print(f"Slide: {record.slide_id} is missing MPP metadata.")
            print(slide.properties)
            missing_ids.append(record.slide_id)
            print("".center(80, "-"))
            continue

    if mppx_values and mppy_values:
        print(
            f"MPP-X: mean={np.mean(mppx_values):.4f}, std={np.std(mppx_values):.4f}, median={np.median(mppx_values):.4f}"
        )
        print(
            f"MPP-Y: mean={np.mean(mppy_values):.4f}, std={np.std(mppy_values):.4f}, median={np.median(mppy_values):.4f}"
        )


if __name__ == "__main__":
    get_mpp_statistics()
