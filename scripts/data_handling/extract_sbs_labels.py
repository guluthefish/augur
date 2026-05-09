"""Extract SBS labels from downloaded data based on the manifest."""

import gzip
import logging
import os
import shutil
import time
from typing import Optional

import pandas as pd
from SigProfilerAssignment import Analyzer
from SigProfilerMatrixGenerator import install as gen_install
from SigProfilerMatrixGenerator.scripts import (
    SigProfilerMatrixGeneratorFunc as mat_gen,
)

from augur.utils.logger import setup_logger

volume = os.environ.get("SIGPROFILERMATRIXGENERATOR_VOLUME")


def ensure_sigprofiler_reference_genome(reference_genome: str) -> None:
    """
    Ensure that the required reference genome is installed for SigProfilerMatrixGenerator.

    Parameters
    ----------
    reference_genome:
        The reference genome to ensure is installed (e.g., "GRCh38").
    """
    if volume is None:
        logging.error(
            "SIGPROFILERMATRIXGENERATOR_VOLUME environment variable is not set. Cannot check or install reference genome."
        )
        raise EnvironmentError(
            "SIGPROFILERMATRIXGENERATOR_VOLUME environment variable is required to check/install reference genome."
        )

    check_path = os.path.join(volume, "tsb", reference_genome)
    if os.path.exists(check_path):
        logging.info(
            "Reference genome '%s' already exists at %s. Skipping installation.",
            reference_genome,
            check_path,
        )
    else:
        logging.info(
            "Reference genome '%s' not found at %s. Installing...",
            reference_genome,
            check_path,
        )
        gen_install.install(reference_genome, rsync=True, volume=volume)


ensure_sigprofiler_reference_genome("GRCh38")


def load_and_filter_manifest(
    manifest_path: str, logger: logging.Logger
) -> pd.DataFrame | None:
    """
    Load the manifest TSV file and filter for entries relevant to SBS assignment extraction.

    Parameters
    ----------
    manifest_path:
        Path to the manifest TSV file containing metadata about downloaded files.
    logger:
        Pre-configured logger for logging warnings and errors during manifest loading and filtering.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only entries relevant for SBS assignment extraction,
        with necessary columns.

    Raises
    ------
    ValueError:
        If the manifest is missing required columns (submitter_id, data_category,
        data_type, filename) or if expected metadata fields are missing for validation.
    """
    t0 = time.perf_counter()
    logger.info(
        "--- STARTING load_and_filter_manifest WITH MANIFEST: %s ---", manifest_path
    )

    manifest_df = pd.read_table(manifest_path, index_col=0, dtype=str)

    required_columns = {"submitter_id", "data_category", "data_type", "filename"}
    missing_columns = required_columns.difference(manifest_df.columns)
    if missing_columns:
        raise ValueError(
            f"Downloaded manifest missing required columns: {sorted(missing_columns)}"
        )

    snv_mask = manifest_df["data_category"].fillna("").eq(
        "Simple Nucleotide Variation"
    ) & manifest_df["data_type"].fillna("").eq("Masked Somatic Mutation")
    snv_manifest = manifest_df.loc[snv_mask].copy()
    if snv_manifest.empty:
        logger.warning(
            "No entries found in manifest for SBS assignment extraction. Manifest path: %s",
            manifest_path,
        )
        return None

    if "subfolder" not in snv_manifest.columns:
        snv_manifest["subfolder"] = (
            snv_manifest["data_category"]
            .fillna("unknown")
            .apply(lambda x: str(x).lower().replace(" ", "_"))
        )

    logger.info(
        "--- END OF load_and_filter_manifest. TIME ELAPSED: %.2fs ---",
        time.perf_counter() - t0,
    )
    return snv_manifest


def prepare_maf_for_label_extraction(
    root_dir: str, manifest_df: pd.DataFrame, logger: logging.Logger
) -> str | None:
    """
    Prepare MAF files for SBS assignment extraction by copying them to a dedicated folder.

    For each entry in `manifest_df`, this function checks for the expected MAF file
    in the ordered_data directory and copies it to <root_dir>/labels/signatures/input_mafs/
    with a standardized filename.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - ordered_data/
          - labels/signatures/input_mafs/
          - logs/
    manifest_df:
        DataFrame containing manifest entries with required columns:
          - submitter_id
          - subfolder
          - filename
    logger:
        Pre-configured logger for logging warnings and errors during file preparation.

    Returns
    -------
    str
        Path to the directory containing prepared MAF files for SBS assignment extraction.

    Raises
    ------
    ValueError:
        If `manifest_df` is missing required columns or if expected metadata fields
        are missing for validation.
    OSError:
        If file copying fails due to OS errors (e.g., permission issues, missing files).
    """
    t0 = time.perf_counter()

    logger.info(
        "STARTING prepare_maf_for_label_extraction WITH %d MANIFEST ENTRIES ---",
        len(manifest_df),
    )

    maf_paths: list[str] = []
    missing = 0
    out_root = os.path.join(root_dir, "labels", "signatures")
    out_maf_dir = os.path.join(out_root, "input_mafs")
    os.makedirs(out_maf_dir, exist_ok=True)

    for _, row in manifest_df.iterrows():
        submitter_id = str(row["submitter_id"]).strip() or "unknown"
        subfolder = str(row["subfolder"]).strip() or "unknown"
        file_id = str(row.name).strip()
        filename = str(row["filename"]).strip()

        src_maf = os.path.join(
            root_dir, "ordered_data", submitter_id, subfolder, file_id, filename
        )
        if os.path.isfile(src_maf):
            dst_maf = os.path.join(out_maf_dir, f"{submitter_id}_{file_id}.maf")
            try:
                if src_maf.endswith(".gz"):
                    with gzip.open(src_maf, "rb") as f_in, open(dst_maf, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(src_maf, dst_maf)
                maf_paths.append(dst_maf)
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to copy MAF file for %s: %s", file_id, src_maf)
        else:
            missing += 1
            logger.warning("Expected MAF file not found for %s: %s", file_id, src_maf)

    logger.info(
        "Prepared %d MAF files for SBS assignment extraction. Missing MAF files for %d entries.",
        len(maf_paths),
        missing,
    )

    logger.info(
        "--- END OF prepare_maf_for_label_extraction. TIME ELAPSED: %.2fs ---",
        time.perf_counter() - t0,
    )

    if not maf_paths:
        return None

    return out_maf_dir


def run_sigprofile(
    root_dir: str, manifest_downloaded: str, logger: Optional[logging.Logger] = None
) -> None:
    """
    Run SigProfiler to extract SBS96 assignment from MAF files specified in the manifest.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - manifests/downloaded/
          - ordered_data/
          - logs/
          - labels/
    manifest_downloaded:
        The manifest filename under <root_dir>/manifests/downloaded/ to process
        for assignment extraction.
    logger:
        Optional pre-configured logger. If None, a default one is created.

    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="extract_sbs_labels")

    t0 = time.perf_counter()
    logger.info(
        "--- STARTING run_sigprofile WITH MANIFEST: %s ---", manifest_downloaded
    )

    # --- Load manifest and filter for SBS-relevant entries ---
    manifest_path = os.path.join(
        root_dir, "manifests", "downloaded", manifest_downloaded
    )
    snv_manifest = load_and_filter_manifest(manifest_path, logger)
    if snv_manifest is None:
        labels_dir = os.path.join(root_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        logger.info("No eligible MAF entries found in manifest.")
        return
    # --- End of manifest loading and filtering ---

    # --- Prepare MAF folder for assignment extraction ---
    maf_dir = prepare_maf_for_label_extraction(root_dir, snv_manifest, logger)
    if maf_dir is None:
        labels_dir = os.path.join(root_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        logger.info("No MAF files prepared for assignment extraction.")
        return

    logger.info(
        "MAF preparation complete. Directory with MAFs ready for assignment extraction: %s",
        maf_dir,
    )
    # --- End of MAF preparation ---

    # --- Generate SBS96 matrix and labels ---
    signature_label_dir = os.path.join(root_dir, "labels", "signatures")
    matrix_dir = os.path.join(signature_label_dir, "matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    _ = mat_gen.SigProfilerMatrixGeneratorFunc(
        project="augur",
        reference_genome="GRCh38",
        path_to_input_files=maf_dir,
        exome=True,
        volume=volume,
        output_directory=matrix_dir,
    )
    logger.info("SBS96 matrix generation complete. Output directory: %s", matrix_dir)
    # --- End of SBS96 matrix generation ---

    # --- Assign COSMIC SBS signatures ---
    assignment_dir = os.path.join(signature_label_dir, "sigprofile_assignment")
    os.makedirs(assignment_dir, exist_ok=True)
    Analyzer.cosmic_fit(
        samples=maf_dir,
        output=assignment_dir,
        genome_build="GRCh38",
        cosmic_version=3.5,
        make_plots=True,
        collapse_to_SBS96=True,
        verbose=True,
        exome=True,
        input_type="vcf",
        context_type="96",
        export_probabilities=True,
        export_probabilities_per_mutation=True,
        volume=volume,
    )
    logger.info(
        "COSMIC SBS signature assignment complete. Output directory: %s",
        assignment_dir,
    )

    logger.info(
        "--- END OF run_sigprofile. TIME ELAPSED: %.2fs ---", time.perf_counter() - t0
    )
    # --- End of COSMIC SBS signature assignment ---


def extract_sbs_labels(
    root_dir: str,
    manifest_downloaded: str,
    min_count: int,
    min_proportion: float,
    top_k_signatures: int,
    logger: Optional[logging.Logger] = None,
) -> tuple[str, str, str]:
    """
    Extract SBS labels from downloaded data based on the manifest.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - manifests/downloaded/
          - ordered_data/
          - logs/
          - labels/
    manifest_downloaded:
        The manifest filename under <root_dir>/manifests/downloaded/ to process
        for label extraction.
    min_count:
        Minimum count threshold for assigning a COSMIC SBS signature label.
    min_proportion:
        Minimum proportion threshold for assigning a COSMIC SBS signature label.
    top_k_signatures:
        Maximum number of COSMIC SBS signature labels to assign per sample
        based on contribution ranking.
    logger:
        Optional pre-configured logger. If None, a default one is created.

    Returns
    -------
    tuple[str, str, str]
        Paths to the generated label files:
        - regression_label_path: Path to the file containing continuous SBS signature contributions.
        - thresholded_multilabel_path: Path to the file containing binary SBS signature labels
          based on thresholds.
        - ranked_multilabel_path: Path to the file containing binary SBS signature labels
          based on top-k ranking.
    """

    logger = logger or setup_logger(
        os.path.join(root_dir, "logs"), name="extract_sbs_labels"
    )
    logger.info(
        "--- STARTING extract_sbs_labels WITH MANIFEST: %s ---", manifest_downloaded
    )

    signature_label_dir = os.path.join(root_dir, "labels", "signatures")
    if not os.path.exists(signature_label_dir):
        logger.error("Signature label directory does not exist.")
        raise FileNotFoundError(
            f"Signature label directory not found: {signature_label_dir}"
        )

    t0 = time.perf_counter()

    #  --- Load SBS activities ---
    assignment_dir = os.path.join(signature_label_dir, "sigprofile_assignment")
    if not os.path.exists(assignment_dir):
        logger.error("SigProfiler assignment directory does not exist.")
        raise FileNotFoundError(
            f"SigProfiler assignment directory not found: {assignment_dir}"
        )

    manifest_path = os.path.join(
        root_dir, "manifests", "downloaded", manifest_downloaded
    )
    snv_manifest = load_and_filter_manifest(manifest_path, logger)
    sample_to_submitter: dict[str, str] = {}
    if snv_manifest is not None:
        for file_id, row in snv_manifest.iterrows():
            submitter_id = str(row["submitter_id"]).strip()
            if submitter_id:
                sample_to_submitter[f"{submitter_id}_{str(file_id).strip()}"] = (
                    submitter_id
                )

    activity_path = os.path.join(
        assignment_dir,
        "Assignment_Solution",
        "Activities",
        "Assignment_Solution_Activities.txt",
    )

    if not os.path.isfile(activity_path):
        logger.error(
            "Expected SigProfiler activities file not found: %s", activity_path
        )
        raise FileNotFoundError(
            f"SigProfiler activities file not found: {activity_path}"
        )

    logger.info("Using activities file for label extraction: %s", activity_path)
    activities = pd.read_csv(activity_path, sep="\t")
    if activities.empty:
        logger.error("Activities file is empty: %s", activity_path)
        raise ValueError(f"Activities file is empty: {activity_path}")

    # --- End of SBS activities loading ---

    # --- Process activities to extract regression labels ---
    signature_cols = [c for c in activities.columns if str(c).startswith("SBS")]
    if not signature_cols:
        logger.error(
            "No signature contribution columns found in activities file: %s",
            activity_path,
        )
        raise ValueError(
            f"No signature columns found in activities file: {activity_path}"
        )

    contributions = activities[["Samples"] + signature_cols].copy()
    contributions.rename(columns={"Samples": "sample_id"}, inplace=True)
    contributions[signature_cols] = (
        contributions[signature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    contributions.insert(
        0,
        "submitter_id",
        contributions["sample_id"]
        .map(sample_to_submitter)
        .fillna(contributions["sample_id"]),
    )
    contributions["submitter_id"] = contributions["submitter_id"].str[:12]
    contributions = contributions[["submitter_id"] + signature_cols]

    eps = 1e-9
    norm_contributions = contributions.copy()
    norm_contributions[signature_cols] = norm_contributions[signature_cols].div(
        norm_contributions[signature_cols].sum(axis=1) + eps, axis=0
    )  # per-sample normalization to get relative contributions

    regression_label_path = os.path.join(
        signature_label_dir, f"sbs_contributions.{manifest_downloaded}"
    )
    norm_contributions.to_csv(regression_label_path, sep="\t", index=False)
    logger.info(
        "SBS regression label extraction complete. Samples processed: %d. Signatures: %d. Output: %s",
        len(norm_contributions),
        len(signature_cols),
        regression_label_path,
    )
    # --- End of regression label extraction ---

    # --- Create binary labels based on threshold ---
    binary = contributions.copy()
    binary[signature_cols] = (contributions[signature_cols] >= min_count) & (
        norm_contributions[signature_cols] >= min_proportion
    )
    binary[signature_cols] = binary[signature_cols].astype(int)

    thresholded_multilabel_path = os.path.join(
        signature_label_dir, f"sbs_labels_thresholded.{manifest_downloaded}"
    )
    binary.to_csv(thresholded_multilabel_path, sep="\t", index=False)
    logger.info(
        "SBS binary label extraction complete with min_count=%d and min_proportion=%.4f. Samples processed: %d. Signatures: %d. Output: %s",
        min_count,
        min_proportion,
        len(binary),
        len(signature_cols),
        thresholded_multilabel_path,
    )
    # --- End of binary label extraction ---

    # --- Create top-N ranked binary labels ---
    ranked_binary = norm_contributions.copy()
    ranked_binary[signature_cols] = ranked_binary[signature_cols].rank(
        method="first", ascending=False, axis=1
    )
    ranked_binary[signature_cols] = (
        ranked_binary[signature_cols] <= top_k_signatures
    ).astype(int)

    ranked_multilabel_path = os.path.join(
        signature_label_dir,
        f"sbs_labels_top{top_k_signatures}.{manifest_downloaded}",
    )
    ranked_binary.to_csv(ranked_multilabel_path, sep="\t", index=False)
    logger.info(
        "SBS top-%d ranked binary label extraction complete. Samples processed: %d. Signatures: %d. Output: %s",
        top_k_signatures,
        len(ranked_binary),
        len(signature_cols),
        ranked_multilabel_path,
    )
    # --- End of top-N ranked binary label extraction ---

    logger.info(
        "--- END OF extract_sbs_labels. TIME ELAPSED: %2fs ---",
        time.perf_counter() - t0,
    )

    return regression_label_path, thresholded_multilabel_path, ranked_multilabel_path
