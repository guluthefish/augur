"""Download PAM50 subtype labels for the TCGA-BRCA cohort from the cBioPortal datahub."""

import logging
import os
from argparse import ArgumentParser
from typing import Optional

import pandas as pd
import yaml

from augur.utils.logger import setup_logger

PAM50_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/refs/heads/master/"
    "public/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
)


def download_pam50(root_dir: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Download PAM50 labels for TCGA-BRCA and write a two-column ``submitter_id``/``subtype`` table.

    Parameters
    ----------
    root_dir:
        Dataset root directory. The labels file is written to
        ``<root_dir>/labels/subtypes/pam50_labels.txt`` and a corresponding
        slide-level atlas is written to ``<root_dir>/atlases/slide_main_atlas.txt``.
    logger:
        Optional pre-configured logger.

    Returns
    -------
    str
        Path to the saved PAM50 labels file.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="download_pam50")

    output_dir = os.path.join(root_dir, "labels", "subtypes")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pam50_labels.txt")

    logger.info("Downloading PAM50 labels from %s", PAM50_URL)
    df = pd.read_csv(PAM50_URL, sep="\t", comment="#", dtype=str)

    missing = {"PATIENT_ID", "SUBTYPE"}.difference(df.columns)
    if missing:
        raise ValueError(f"PAM50 source is missing expected columns: {sorted(missing)}")

    df = df.rename(columns={"PATIENT_ID": "submitter_id", "SUBTYPE": "subtype"})
    df = df[["submitter_id", "subtype"]].set_index("submitter_id")
    df.to_csv(output_file, sep="\t")
    logger.info("Saved %d PAM50 labels to %s", len(df), output_file)

    atlas_dir = os.path.join(root_dir, "atlases")
    os.makedirs(atlas_dir, exist_ok=True)
    atlas_path = os.path.join(atlas_dir, "slide_main_atlas.txt")
    atlas_df = pd.DataFrame({"type": ["subtyping"], "path": [output_file]})
    atlas_df.to_csv(atlas_path, sep="\t", index=False)
    logger.info("Wrote slide main atlas to %s", atlas_path)

    return output_file


def main():
    """Run the PAM50 downloader from a YAML config file."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to data config file.")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please provide --config for the PAM50 downloader.")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "root_dir" not in config:
        raise KeyError("Config file is missing required key: root_dir")

    download_pam50(root_dir=os.path.expanduser(str(config["root_dir"])))


if __name__ == "__main__":
    main()
