"""Main script for data handling."""

from argparse import ArgumentParser
import os
import pandas as pd
import yaml

from .download_bcss import download_bcss
from .download_data import download_data
from .extract_sbs_labels import (
    extract_sbs_labels,
    run_sigprofile,
)
from .process_manifests import (
    create_ready_for_download_manifest,
    filter_out_already_downloaded,
    split_manifest,
)
from .remove_defects import remove_missing_mpp_metadata
from .reorder_data import reorder_data


def main():
    """
    Download and process data.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to data config file.")
    _args = parser.parse_args()
    config_path = _args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Check required config keys ---
    root_dir = config["root_dir"]
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(
            f"root_dir does not exist or is not a directory: {root_dir}"
        )

    manifests_raw = config["manifests_raw"]
    raw_dir = os.path.join(root_dir, "manifests", "raw")
    for f in manifests_raw:
        p = os.path.join(raw_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing manifest: {p}")

    manifests_for_download = config["manifests_for_download"]
    if manifests_for_download is not False:
        ready_dir = os.path.join(root_dir, "manifests", "ready_for_download")
        for f in manifests_for_download:
            p = os.path.join(ready_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing manifest for download: {p}")

    n_files_per_command = config["n_files_per_command"]
    if not isinstance(n_files_per_command, int) or n_files_per_command <= 0:
        raise ValueError("n_files_per_command must be a positive integer")

    tcga_user_token_file = config["tcga_user_token_file"]
    if tcga_user_token_file is not False:
        token_path = os.path.expanduser(tcga_user_token_file)
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Token file not found: {token_path}")
    else:
        token_path = None

    n_processes = config["n_processes"]
    if not isinstance(n_processes, int) or n_processes <= 0:
        raise ValueError("n_processes must be a positive integer")

    min_count = config["min_count"]
    if not isinstance(min_count, int) or min_count < 0:
        raise ValueError("min_count must be a non-negative integer")

    min_proportion = config["min_proportion"]
    if not isinstance(min_proportion, (int, float)) or not 0 <= min_proportion <= 1:
        raise ValueError("min_proportion must be a number between 0 and 1")

    top_k_accuracy_score = config["top_k_signatures"]
    if not isinstance(top_k_accuracy_score, int) or top_k_accuracy_score <= 0:
        raise ValueError("top_k_signatures must be a positive integer")

    bcss_mpp = config["bcss_mpp"]
    if not isinstance(bcss_mpp, (int, float)) or bcss_mpp <= 0:
        raise ValueError("bcss_mpp must be a positive number")

    bcss_pipelines = config["bcss_pipelines"]
    if not isinstance(bcss_pipelines, list) or not all(
        isinstance(p, str) for p in bcss_pipelines
    ):
        raise ValueError("bcss_pipelines must be a list of strings")
    # --- End of config checks ---

    # --- Process manifests ---
    if manifests_for_download is False:
        manifests_merged = create_ready_for_download_manifest(
            root_dir=root_dir, manifests_raw=manifests_raw
        )
    else:
        manifests_merged = manifests_for_download

    manifest_filtered = filter_out_already_downloaded(
        root_dir=root_dir, manifests_merged=manifests_merged
    )

    manifests_splitted = split_manifest(
        root_dir=root_dir,
        manifest_filtered=manifest_filtered,
        n_files_per_command=n_files_per_command,
    )
    # --- End of manifest processing ---

    # --- Download and reorder data ---
    download_data(
        root_dir=root_dir,
        manifests_ready=manifests_splitted,
        n_processes=n_processes,
        tcga_user_token_file=token_path,
    )

    manifest_downloaded = reorder_data(
        root_dir=root_dir,
        manifests_ready=manifests_merged,
    )
    # --- End of data downloading and reordering ---

    # --- Remove defective samples ---
    manifest_downloaded = remove_missing_mpp_metadata(
        root_dir=root_dir, manifest_downloaded=manifest_downloaded
    )
    # --- End of defective sample removal ---

    # # --- Extract SBS labels ---
    # run_sigprofile(
    #     root_dir=root_dir,
    #     manifest_downloaded=manifest_downloaded,
    # )

    # regression_label_path, thresholded_multilabel_path, ranked_multilabel_path = (
    #     extract_sbs_labels(
    #         root_dir=root_dir,
    #         manifest_downloaded=manifest_downloaded,
    #         min_count=min_count,
    #         min_proportion=min_proportion,
    #         top_k_signatures=top_k_accuracy_score,
    #     )
    # )
    # # --- End of SBS label extraction ---

    # --- Create atlases ---
    atlas_dir = os.path.join(root_dir, "atlases")
    os.makedirs(atlas_dir, exist_ok=True)

    # slide_pretext_atlas_path = os.path.join(atlas_dir, "slide_pretext_atlas.txt")
    # sbs_label_dir = os.path.join(root_dir, "labels", "signatures")
    # slide_pretext_atlas = {
    #     "sbs_regression": os.path.join(sbs_label_dir, regression_label_path),
    #     "sbs_thresholded_multilabel": os.path.join(
    #         sbs_label_dir, thresholded_multilabel_path
    #     ),
    #     "sbs_ranked_multilabel": os.path.join(sbs_label_dir, ranked_multilabel_path),
    # }
    # slide_pretext_atlas_df = pd.DataFrame.from_dict(
    #     slide_pretext_atlas, orient="index", columns=["path"]
    # )
    # slide_pretext_atlas_df.index.name = "type"
    # slide_pretext_atlas_df.to_csv(slide_pretext_atlas_path, sep="\t", index=True, header=True)

    manifest_atlas_path = os.path.join(atlas_dir, "manifest_atlas.txt")
    manifest_atlas = {
        "manifest_downloaded": os.path.join(
            root_dir, "manifests", "downloaded", manifest_downloaded
        )
    }
    manifest_atlas_df = pd.DataFrame.from_dict(
        manifest_atlas, orient="index", columns=["path"]
    )
    manifest_atlas_df.index.name = "type"
    manifest_atlas_df.to_csv(manifest_atlas_path, sep="\t", index=True, header=True)
    # --- End of atlas creation ---

    # # --- Download BCSS data ---
    # download_bcss(root_dir, bcss_mpp, bcss_pipelines)
    # # --- End of BCSS data downloading ---


if __name__ == "__main__":
    main()
