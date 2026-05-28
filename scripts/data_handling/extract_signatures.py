"""Extract SBS labels from downloaded data based on the manifest."""

import argparse
import gzip
import logging
import os
import shutil
from typing import Optional

import pandas as pd
from SigProfilerAssignment import Analyzer
from SigProfilerMatrixGenerator import install as gen_install
from SigProfilerMatrixGenerator.scripts import (
    CNVMatrixGenerator as scna,
    SigProfilerMatrixGeneratorFunc as mat_gen,
)
import yaml

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


SIGNATURE_INPUT_FILTERS: dict[str, tuple[str, str]] = {
    "snv": ("Simple Nucleotide Variation", "Masked Somatic Mutation"),
    "cnv": ("Copy Number Variation", "Allele-specific Copy Number Segment"),
}


def load_and_filter_manifest(
    manifest_path: str, logger: logging.Logger
) -> dict[str, pd.DataFrame]:
    """
    Load the manifest TSV file and split entries by signature input type.

    Parameters
    ----------
    manifest_path:
        Path to the manifest TSV file containing metadata about downloaded files.
    logger:
        Pre-configured logger for logging warnings and errors during manifest loading and filtering.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from input-type key (see ``SIGNATURE_INPUT_FILTERS``) to the filtered
        manifest rows for that type. Types with no matching rows are omitted.

    Raises
    ------
    ValueError:
        If the manifest is missing required columns (submitter_id, data_category,
        data_type, filename).
    """
    manifest_df = pd.read_table(manifest_path, index_col=0, dtype=str)

    required_columns = {"submitter_id", "data_category", "data_type", "filename"}
    missing_columns = required_columns.difference(manifest_df.columns)
    if missing_columns:
        raise ValueError(
            f"Downloaded manifest missing required columns: {sorted(missing_columns)}"
        )

    result: dict[str, pd.DataFrame] = {}
    for key, (category, dtype) in SIGNATURE_INPUT_FILTERS.items():
        mask = manifest_df["data_category"].fillna("").eq(category) & manifest_df[
            "data_type"
        ].fillna("").eq(dtype)
        sub = manifest_df.loc[mask].copy()
        if sub.empty:
            logger.warning(
                "No '%s' entries (%s / %s) found in manifest: %s",
                key,
                category,
                dtype,
                manifest_path,
            )
            continue

        if "subfolder" not in sub.columns:
            sub["subfolder"] = (
                sub["data_category"]
                .fillna("unknown")
                .apply(lambda x: str(x).lower().replace(" ", "_"))
            )
        result[key] = sub
        logger.info("Loaded %d '%s' entries from manifest.", len(sub), key)

    return result


INPUT_SUBDIRS: dict[str, tuple[str, str]] = {
    "snv": ("input_mafs", "maf"),
    "cnv": ("input_cnv", "tsv"),
}


def _copy_input_file(src: str, dst: str, logger: logging.Logger) -> bool:
    """Copy ``src`` to ``dst``, transparently decompressing if ``src`` ends in ``.gz``."""
    try:
        if src.endswith(".gz"):
            with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dst)
        return True
    except Exception:  # pylint: disable=broad-except
        logger.exception("Failed to copy input file: %s -> %s", src, dst)
        return False


def prepare_inputs_for_label_extraction(
    root_dir: str,
    manifests: dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Prepare per-sample input files for signature assignment.

    For each input type in ``manifests``, copy the corresponding source files from
    ``ordered_data/`` into a type-specific directory under ``labels/signatures/``
    (see ``INPUT_SUBDIRS``), with standardized ``<submitter>_<file_id>.<ext>`` names.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing ``ordered_data/`` and ``labels/signatures/``.
    manifests:
        Mapping from input-type key (see ``SIGNATURE_INPUT_FILTERS``) to the filtered
        manifest rows for that type. Each DataFrame must include the columns
        ``submitter_id``, ``subfolder``, and ``filename``, and use the file UUID
        as its index.
    logger:
        Pre-configured logger for logging warnings and errors during file preparation.

    Returns
    -------
    dict[str, str]
        Mapping from input-type key to the directory containing prepared files.
        Types for which no files could be prepared are omitted.
    """
    out_root = os.path.join(root_dir, "labels", "signatures")
    prepared: dict[str, str] = {}

    for key, manifest_df in manifests.items():
        if key not in INPUT_SUBDIRS:
            logger.warning("No input subdir configured for type '%s'. Skipping.", key)
            continue
        subdir_name, extension = INPUT_SUBDIRS[key]
        out_dir = os.path.join(out_root, subdir_name)
        os.makedirs(out_dir, exist_ok=True)

        copied = 0
        missing = 0
        for _, row in manifest_df.iterrows():
            submitter_id = str(row["submitter_id"]).strip() or "unknown"
            subfolder = str(row["subfolder"]).strip() or "unknown"
            file_id = str(row.name).strip()
            filename = str(row["filename"]).strip()

            src = os.path.join(
                root_dir, "ordered_data", submitter_id, subfolder, file_id, filename
            )
            if not os.path.isfile(src):
                missing += 1
                logger.warning(
                    "Expected '%s' input file not found for %s: %s",
                    key,
                    file_id,
                    src,
                )
                continue

            dst = os.path.join(out_dir, f"{submitter_id}_{file_id}.{extension}")
            if _copy_input_file(src, dst, logger):
                copied += 1

        logger.info(
            "Prepared %d '%s' input files for signature assignment. Missing for %d entries.",
            copied,
            key,
            missing,
        )
        if copied:
            prepared[key] = out_dir

    return prepared


SNV_FIT_CONFIGS: list[tuple[str, str, dict]] = [
    ("sbs", "96", {"collapse_to_SBS96": True}),
    ("dbs", "DINUC", {}),
    ("id", "ID", {}),
]


def _merge_cnv_inputs(
    cnv_input_dir: str, output_dir: str, logger: logging.Logger
) -> str | None:
    """
    Concatenate per-sample TCGA allele-specific CNV segment TSVs into a single file
    suitable for ``scna.generateCNVMatrix(file_type="TCGA", ...)``.

    The standardized ``<submitter>_<file_id>`` filename (produced by
    :func:`prepare_inputs_for_label_extraction`) is written into the ``GDC_Aliquot``
    column so CN samples align with SNV samples downstream.
    """
    frames: list[pd.DataFrame] = []
    for entry in sorted(os.listdir(cnv_input_dir)):
        if not entry.endswith(".tsv"):
            continue
        path = os.path.join(cnv_input_dir, entry)
        try:
            seg = pd.read_table(path, dtype=str)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to read CNV input: %s", path)
            continue
        if seg.empty:
            continue

        sample_id = os.path.splitext(entry)[0]
        if "GDC_Aliquot" in seg.columns:
            seg["GDC_Aliquot"] = sample_id
        else:
            seg.insert(0, "GDC_Aliquot", sample_id)
        frames.append(seg)

    if not frames:
        logger.warning("No CNV inputs found in %s", cnv_input_dir)
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged_path = os.path.join(output_dir, "cnv_segments.tsv")
    merged.to_csv(merged_path, sep="\t", index=False)
    logger.info(
        "Merged %d CNV input files into %s (%d total segments).",
        len(frames),
        merged_path,
        len(merged),
    )
    return merged_path


def run_sigprofile(
    root_dir: str, final_manifest: str, logger: Optional[logging.Logger] = None
) -> None:
    """
    Run SigProfiler to extract COSMIC SBS, DBS, ID, and CN signature assignments
    from the inputs specified in the manifest.

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - manifests/downloaded/
          - ordered_data/
          - logs/
          - labels/
    final_manifest:
        The manifest filename under ``<root_dir>/manifests/downloaded/`` to process
        for assignment extraction.
    logger:
        Optional pre-configured logger. If None, a default one is created.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger or setup_logger(log_dir, name="extract_signature")

    # --- Load manifest and split by input type ---
    manifest_path = os.path.join(root_dir, "manifests", "downloaded", final_manifest)
    manifests = load_and_filter_manifest(manifest_path, logger)
    if not manifests:
        labels_dir = os.path.join(root_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        logger.info("No eligible signature entries found in manifest.")
        return
    # --- End of manifest loading ---

    # --- Prepare per-type input directories ---
    input_dirs = prepare_inputs_for_label_extraction(root_dir, manifests, logger)
    if not input_dirs:
        labels_dir = os.path.join(root_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        logger.info("No input files prepared for signature assignment.")
        return
    logger.info("Input preparation complete: %s", input_dirs)
    # --- End of input preparation ---

    signature_label_dir = os.path.join(root_dir, "labels", "signatures")
    matrix_dir = os.path.join(signature_label_dir, "matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    assignment_root = os.path.join(signature_label_dir, "sigprofile_assignment")
    os.makedirs(assignment_root, exist_ok=True)

    # --- SBS / DBS / ID from MAFs ---
    if "snv" in input_dirs:
        maf_dir = input_dirs["snv"]
        _ = mat_gen.SigProfilerMatrixGeneratorFunc(
            project="augur",
            reference_genome="GRCh38",
            path_to_input_files=maf_dir,
            exome=True,
            volume=volume,
            output_directory=matrix_dir,
        )
        logger.info(
            "SBS/DBS/ID matrix generation complete. Output directory: %s", matrix_dir
        )

        for sig_name, context_type, extra in SNV_FIT_CONFIGS:
            out = os.path.join(assignment_root, sig_name)
            os.makedirs(out, exist_ok=True)
            Analyzer.cosmic_fit(
                samples=maf_dir,
                output=out,
                genome_build="GRCh38",
                cosmic_version=3.5,
                make_plots=True,
                verbose=True,
                exome=True,
                input_type="vcf",
                context_type=context_type,
                export_probabilities=True,
                export_probabilities_per_mutation=True,
                volume=volume,
                **extra,
            )
            logger.info(
                "COSMIC %s signature assignment complete. Output directory: %s",
                sig_name.upper(),
                out,
            )
    # --- End of SBS/DBS/ID assignment ---

    # --- CN48 from allele-specific CNV segments ---
    if "cnv" in input_dirs:
        merged_cnv_path = _merge_cnv_inputs(input_dirs["cnv"], matrix_dir, logger)
        if merged_cnv_path is None:
            logger.warning("Skipping CN signature assignment: no usable CNV inputs.")
        else:
            cnv_matrix_dir = os.path.join(matrix_dir, "CNV")
            os.makedirs(cnv_matrix_dir, exist_ok=True)
            scna.generateCNVMatrix("TCGA", merged_cnv_path, "augur", cnv_matrix_dir)
            cnv_matrix_path = os.path.join(cnv_matrix_dir, "augur.CNV48.matrix")
            logger.info("CN48 matrix generation complete. Output: %s", cnv_matrix_path)

            cnv_assignment_dir = os.path.join(assignment_root, "cnv")
            os.makedirs(cnv_assignment_dir, exist_ok=True)
            Analyzer.cosmic_fit(
                samples=cnv_matrix_path,
                output=cnv_assignment_dir,
                genome_build="GRCh38",
                cosmic_version=3.5,
                make_plots=True,
                verbose=True,
                input_type="matrix",
                context_type="CNV48",
                export_probabilities=True,
                volume=volume,
            )
            logger.info(
                "COSMIC CN signature assignment complete. Output directory: %s",
                cnv_assignment_dir,
            )
    # --- End of CN48 assignment ---


SIGNATURE_LABEL_CONFIGS: list[tuple[str, str]] = [
    ("sbs", "SBS"),
    ("dbs", "DBS"),
    ("id", "ID"),
    ("cnv", "CN"),
]


def extract_signature_labels(
    root_dir: str,
    final_manifest: str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, str]:
    """
    Build per-sample normalized signature contribution tables for each signature
    class assigned by :func:`run_sigprofile` (SBS, DBS, ID, CN).

    One TSV is written per class to
    ``<root_dir>/labels/signatures/<class>_contributions.<final_manifest>``,
    with columns ``submitter_id`` plus one column per COSMIC signature, where each
    row sums to 1 (per-sample normalization).

    Parameters
    ----------
    root_dir:
        Root dataset directory containing:
          - manifests/downloaded/
          - logs/
          - labels/signatures/sigprofile_assignment/
    final_manifest:
        The manifest filename under ``<root_dir>/manifests/downloaded/`` used to
        recover the sample -> submitter mapping and to suffix the output filenames.
    logger:
        Optional pre-configured logger. If None, a default one is created.

    Returns
    -------
    dict[str, str]
        Mapping from signature class key (e.g. ``"sbs"``) to the path of the
        written contribution table. Classes without an assignment directory or
        without matching signature columns are skipped.
    """
    logger = logger or setup_logger(
        os.path.join(root_dir, "logs"), name="extract_signature"
    )

    signature_label_dir = os.path.join(root_dir, "labels", "signatures")
    assignment_root = os.path.join(signature_label_dir, "sigprofile_assignment")
    if not os.path.isdir(assignment_root):
        raise FileNotFoundError(
            f"SigProfiler assignment root not found: {assignment_root}"
        )

    # Recover sample_id -> submitter_id from the manifest (sample_id is the
    # standardized "<submitter>_<file_id>" written by prepare_inputs_for_label_extraction
    # and used as the "Samples" / "GDC_Aliquot" column by SigProfiler).
    manifest_path = os.path.join(root_dir, "manifests", "downloaded", final_manifest)
    manifests = load_and_filter_manifest(manifest_path, logger)
    sample_to_submitter: dict[str, str] = {}
    for sub_manifest in manifests.values():
        for file_id, row in sub_manifest.iterrows():
            submitter_id = str(row["submitter_id"]).strip()
            if submitter_id:
                sample_to_submitter[f"{submitter_id}_{str(file_id).strip()}"] = (
                    submitter_id
                )

    written: dict[str, str] = {}
    for sig_class, col_prefix in SIGNATURE_LABEL_CONFIGS:
        activity_path = os.path.join(
            assignment_root,
            sig_class,
            "Assignment_Solution",
            "Activities",
            "Assignment_Solution_Activities.txt",
        )
        if not os.path.isfile(activity_path):
            logger.warning(
                "No activities file for '%s' signatures: %s. Skipping.",
                sig_class,
                activity_path,
            )
            continue

        activities = pd.read_csv(activity_path, sep="\t")
        if activities.empty:
            logger.warning("Activities file empty: %s. Skipping.", activity_path)
            continue

        signature_cols = [
            c for c in activities.columns if str(c).startswith(col_prefix)
        ]
        if not signature_cols:
            logger.warning(
                "No '%s*' signature columns found in %s. Skipping.",
                col_prefix,
                activity_path,
            )
            continue

        contributions = activities[["Samples"] + signature_cols].copy()
        contributions.rename(columns={"Samples": "sample_id"}, inplace=True)
        contributions[signature_cols] = (
            contributions[signature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
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
        contributions[signature_cols] = contributions[signature_cols].div(
            contributions[signature_cols].sum(axis=1) + eps, axis=0
        )

        out_path = os.path.join(
            signature_label_dir,
            f"{sig_class}_contributions.txt",
        )
        contributions.to_csv(out_path, sep="\t", index=False)
        logger.info(
            "%s contribution table written: %d samples, %d signatures -> %s",
            sig_class.upper(),
            len(contributions),
            len(signature_cols),
            out_path,
        )
        written[sig_class] = out_path

    if not written:
        logger.warning("No signature contribution tables were written.")
        return written

    # --- Upsert atlas entries for the regression label files ---
    atlas_dir = os.path.join(root_dir, "atlases")
    os.makedirs(atlas_dir, exist_ok=True)
    atlas_path = os.path.join(atlas_dir, "slide_subtask_atlas.txt")

    if os.path.isfile(atlas_path):
        atlas_df = pd.read_table(atlas_path, dtype=str)
        if not {"type", "path"}.issubset(atlas_df.columns):
            raise ValueError(
                f"Existing atlas {atlas_path} missing required columns 'type' and 'path'."
            )
    else:
        atlas_df = pd.DataFrame(columns=["type", "path"])

    new_rows = pd.DataFrame(
        [
            {"type": f"{sig_class}_regression", "path": path}
            for sig_class, path in written.items()
        ]
    )
    atlas_df = atlas_df[~atlas_df["type"].isin(new_rows["type"])]
    atlas_df = pd.concat([atlas_df, new_rows], ignore_index=True)
    atlas_df.to_csv(atlas_path, sep="\t", index=False)
    logger.info(
        "Updated slide_subtask atlas with %d regression entries: %s",
        len(new_rows),
        atlas_path,
    )
    # --- End of atlas update ---

    return written


def main():
    """
    CLI entry: extract COSMIC SBS, DBS, ID, and CN signature assignments and
    write per-class normalized contribution tables, driven by a YAML config.

    Required config keys:
        root_dir: dataset root containing ``manifests/``, ``ordered_data/``,
                  ``atlases/manifest_atlas.txt`` (with a ``final_manifest``
                  entry pointing at the downloaded manifest).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract COSMIC SBS, DBS, ID, and CN signature assignments from "
            "downloaded data based on the manifest."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file specifying root_dir.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root_dir = config["root_dir"]
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(
            f"root_dir does not exist or is not a directory: {root_dir}"
        )

    atlas_path = os.path.join(root_dir, "atlases", "manifest_atlas.txt")
    if not os.path.isfile(atlas_path):
        raise FileNotFoundError(f"Manifest atlas not found: {atlas_path}")

    atlas_df = pd.read_table(atlas_path, dtype=str)
    matches = atlas_df[atlas_df["type"] == "final_manifest"]
    if matches.empty:
        raise ValueError(f"Atlas {atlas_path} is missing a 'final_manifest' entry.")
    manifest_path = str(matches.iloc[0]["path"]).strip()
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest file referenced by atlas not found: {manifest_path}"
        )

    final_manifest = os.path.basename(manifest_path)
    logger = setup_logger(os.path.join(root_dir, "logs"), name="extract_signature")
    logger.info("Loaded config %s; manifest: %s", args.config, manifest_path)

    run_sigprofile(root_dir, final_manifest, logger=logger)
    extract_signature_labels(root_dir, final_manifest, logger=logger)


if __name__ == "__main__":
    main()
