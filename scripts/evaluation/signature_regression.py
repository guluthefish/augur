"""Top-k signature-set agreement for slide-level mutational-signature regression.

Reads one or more ``<subtask>.txt`` prediction tables produced by
``scripts/evaluation/slide_results.py`` (via ``cloud/07_predict_slide.sh``).
Each file is a TSV with columns::

    slide_id  <SIG_0>  <SIG_1>  ...  <SIG_n>

where ``slide_id`` is the GDC ``file_id`` and the trailing columns are the
model's predicted COSMIC signature exposures (e.g. ``SBS1``, ``SBS2``, ...).

Unlike ``subtyping.txt``, these files carry NO ground truth, so each slide is
joined to its true exposure vector exactly the way the training datamodule
labels it::

    file_id  --(manifest)-->  submitter_id  --(labels)-->  true exposure vector

Ground truth is the COSMIC contribution table loaded with
``augur.datasets.mutational_signature.load_signature_labels`` (see that module
for the label format), resolved through ``<root_dir>/atlases/``:
``slide_subtask_atlas.txt`` gives the contribution TSV for the subtask, and
``manifest_atlas.txt`` gives the GDC manifest that maps each ``file_id`` to its
``submitter_id``.

Metric (per slide): take the top-m signatures by TRUE exposure (set ``T``) and
the top-m by PREDICTED exposure (set ``P``), then report

  * IoU / Jaccard = ``|T ∩ P| / |T ∪ P|``           -- the headline metric,
  * overlap@k     = ``|T ∩ P| / m``                 -- = precision@k = recall@k
                                                       since ``|T| = |P| = m``.

Both are scale-invariant: they compare only the *sets* of dominant signatures,
which is what we want -- the regression head's output scale need not match the
label scale, only the ranking of the dominant signatures.

``m = min(k, n_active)`` where ``n_active`` is the slide's number of strictly
positive true exposures. COSMIC profiles are sparse (most slides have only a
handful of active signatures, far fewer than a typical k), so a *fixed* top-k
would pad the true set with zero-exposure ties and credit spurious matches --
a signal-free predictor would then score a high IoU purely from tie-filler
collisions. Capping k to the active count keeps both sets to genuinely active
top signatures. Consequently, for slides with ``n_active < k`` the top-k and a
larger top-k' collapse to the same value (there is nothing more to recover),
and slides with no active signature are skipped.

Two k-independent metrics are always reported alongside the top-k table. The
subtask head trains with ``KL(target || softmax(prediction))``, so the raw
predicted columns are logits and ``softmax(prediction)`` is the predicted
exposure distribution (this does not affect the top-k sets, since softmax is
monotone within a slide):

  * total variation = ``0.5 * |softmax(pred) - true|_1`` per slide (0 = identical
    profile, 1 = disjoint; lower is better), averaged over slides;
  * macro AUROC (active vs not) = per-signature one-vs-rest AUROC of "is this
    signature active (true > 0)" scored by predicted exposure, macro-averaged
    over signatures present in both classes. It ranks *patients*, so the
    marginal/prevalence predictor scores exactly 0.5 -- that is the chance line.

With ``--baselines`` the same total variation and macro AUROC are also reported
for the prevalence predictor (the cohort-mean profile) and the random predictor
(the uniform distribution): prevalence TV is the marginal bar the model must
beat, and both baselines' macro AUROC is 0.5 by construction.

Passing several files (e.g. one per CV fold, like the AUROC script) pools their
slides. The script reports the pooled mean (every slide weighted equally) and
the per-fold mean ± std (the unweighted across-fold average), so both readings
of "the average over folds" are available.

Examples
--------
    # one fold
    python scripts/evaluation/signature_regression.py \
        outputs/evaluation/clam-full-mb-gated-resnet50-full-fold0/sbs_regression.txt

    # pool all 5 folds, top-5 and top-10, with bootstrap CIs and the
    # prevalence/random baselines, against the production label root
    python scripts/evaluation/signature_regression.py \
        outputs/evaluation/clam-full-mb-gated-resnet50-full-fold*/sbs_regression.txt \
        --root-dir data/TCGA-BRCA --top-k 5 10 --bootstrap 1000 --baselines
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# augur is installed (editable), but add the repo root defensively so the
# import works even from a checkout that was never pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augur.datasets.mutational_signature import load_signature_labels  # noqa: E402

_SLIDE_ID_COLUMN = "slide_id"


# ---------------------------------------------------------------------------
# Loading: predictions, the file_id -> submitter_id map, and the labels
# ---------------------------------------------------------------------------
def signature_columns(frame: pd.DataFrame) -> list[str]:
    """Predicted-signature columns = everything that isn't ``slide_id``."""
    return [column for column in frame.columns if column != _SLIDE_ID_COLUMN]


def load_prediction_frames(paths: Sequence[str]) -> list[tuple[str, pd.DataFrame]]:
    """Read each prediction TSV, keeping its path (fold identity) alongside.

    All files must share the same signature columns so they can be pooled and
    so a single label vector aligns to every fold.
    """
    frames: list[tuple[str, pd.DataFrame]] = []
    columns: tuple[str, ...] | None = None
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        if _SLIDE_ID_COLUMN not in frame.columns:
            raise ValueError(f"{path} has no '{_SLIDE_ID_COLUMN}' column.")
        signature_cols = tuple(signature_columns(frame))
        if not signature_cols:
            raise ValueError(f"{path} has no signature columns.")
        if columns is None:
            columns = signature_cols
        elif signature_cols != columns:
            raise ValueError(
                f"Signature-column mismatch in {path}.\n  expected: {columns}\n"
                f"  found:    {signature_cols}"
            )
        frames.append((path, frame))
    return frames


def infer_subtask(paths: Sequence[str], override: str | None) -> str:
    """Subtask name (atlas key), inferred from the file stem unless overridden.

    The prediction files are named ``<subtask>.txt`` (e.g. ``sbs_regression``),
    so the stem is the atlas key. All pooled files must agree.
    """
    if override is not None:
        return override
    stems = {os.path.splitext(os.path.basename(path))[0] for path in paths}
    if len(stems) != 1:
        raise ValueError(
            f"Could not infer a single subtask from {sorted(stems)}; pass "
            "--subtask explicitly."
        )
    return stems.pop()


def _resolve_from_atlas(atlas_path: str, key: str) -> str:
    """Resolve a path from a ``type``/``path`` atlas TSV (see augur.datasets.utils).

    Replicated locally (rather than imported) because ``augur.datasets.utils``
    pulls in torch/openslide/cv2 at import time, which this numpy+pandas metric
    script must not require.
    """
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    atlas = pd.read_table(atlas_path, index_col=0, dtype=str)
    if key not in atlas.index:
        raise ValueError(f"Atlas {atlas_path} is missing the '{key}' entry.")
    resolved = str(atlas.loc[key, "path"]).strip()
    if not resolved or not os.path.exists(resolved):
        raise FileNotFoundError(f"Resolved path for '{key}' not found: {resolved}")
    return resolved


def resolve_labels_path(root_dir: str, subtask: str, override: str | None) -> str:
    """Path to the subtask's COSMIC contribution TSV."""
    if override is not None:
        if not os.path.exists(override):
            raise FileNotFoundError(f"Labels file not found: {override}")
        return override
    atlas_path = os.path.join(root_dir, "atlases", "slide_subtask_atlas.txt")
    return _resolve_from_atlas(atlas_path, subtask)


def resolve_manifest_path(root_dir: str, override: str | None) -> str:
    """Path to the GDC manifest TSV (carries file_id -> submitter_id)."""
    if override is not None:
        if not os.path.exists(override):
            raise FileNotFoundError(f"Manifest file not found: {override}")
        return override
    atlas_path = os.path.join(root_dir, "atlases", "manifest_atlas.txt")
    return _resolve_from_atlas(atlas_path, "final_manifest")


def load_file_to_submitter(manifest_path: str) -> dict[str, str]:
    """Build ``file_id -> submitter_id`` from the manifest (index = ``id``)."""
    manifest = pd.read_table(manifest_path, index_col=0, dtype=str)
    if "submitter_id" not in manifest.columns:
        raise ValueError(f"Manifest {manifest_path} has no 'submitter_id' column.")
    submitters = manifest["submitter_id"].astype(str).str.strip()
    return {
        str(file_id).strip(): submitter for file_id, submitter in submitters.items()
    }


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def _topk_indices(vector: np.ndarray, k: int) -> set[int]:
    """Indices of the ``k`` largest entries; ties broken by signature index."""
    # Negate for descending order; mergesort is stable so ties resolve to the
    # lower (earlier) signature index deterministically.
    return set(int(i) for i in np.argsort(-vector, kind="mergesort")[:k])


def _gather_evaluable(
    frame: pd.DataFrame,
    signature_cols: list[str],
    align_index: np.ndarray,
    file_to_submitter: dict[str, str],
    labels_by_submitter: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Join one frame's slides to their labels and return evaluable rows.

    Returns ``(trues, preds, n_active, counts)`` where ``trues`` / ``preds`` are
    ``(N, S)`` arrays (true exposure + model prediction, aligned to
    ``signature_cols``) over the ``N`` slides that map to a submitter with a
    signature label AND have at least one active (strictly positive) true
    exposure; ``n_active`` is the per-slide active count; ``counts`` tallies the
    skips. Returning the raw arrays lets the baselines reuse the exact same
    slide set and active-cap logic.
    """
    predictions = frame[signature_cols].to_numpy(dtype=np.float64)
    slide_ids = frame[_SLIDE_ID_COLUMN].astype(str).tolist()
    num_signatures = len(signature_cols)

    trues: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    actives: list[int] = []
    counts = {
        "evaluated": 0,
        "missing_submitter": 0,
        "missing_label": 0,
        "no_active_signatures": 0,
        "active_total": 0,
    }

    for row_index, slide_id in enumerate(slide_ids):
        submitter = file_to_submitter.get(slide_id)
        if submitter is None:
            counts["missing_submitter"] += 1
            continue
        label_vector = labels_by_submitter.get(submitter)
        if label_vector is None:
            counts["missing_label"] += 1
            continue

        true_vector = label_vector[align_index]
        # Only strictly-positive exposures are genuinely active signatures.
        n_active = int(np.count_nonzero(true_vector > 0.0))
        if n_active == 0:
            counts["no_active_signatures"] += 1
            continue

        counts["evaluated"] += 1
        counts["active_total"] += n_active
        trues.append(true_vector)
        preds.append(predictions[row_index])
        actives.append(n_active)

    trues_arr = np.asarray(trues, dtype=np.float64).reshape(-1, num_signatures)
    preds_arr = np.asarray(preds, dtype=np.float64).reshape(-1, num_signatures)
    actives_arr = np.asarray(actives, dtype=np.int64)
    return trues_arr, preds_arr, actives_arr, counts


def _topk_metrics(
    trues: np.ndarray,
    preds: np.ndarray,
    n_active: np.ndarray,
    ks: Sequence[int],
    num_signatures: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-slide top-k IoU + overlap@k for aligned ``(trues, preds)`` arrays.

    ``m = min(k, n_active, S)`` per slide, so the true set holds only active
    signatures. Used for both the model predictions and the prevalence baseline
    (which passes the cohort-mean profile, broadcast, as ``preds``).
    """
    per_k: dict[int, dict[str, list[float]]] = {
        k: {"iou": [], "overlap": []} for k in ks
    }
    for row in range(trues.shape[0]):
        true_vector = trues[row]
        pred_vector = preds[row]
        active = int(n_active[row])
        for k in ks:
            m = min(k, active, num_signatures)
            true_top = _topk_indices(true_vector, m)
            pred_top = _topk_indices(pred_vector, m)
            intersection = len(true_top & pred_top)
            union = len(true_top | pred_top)
            per_k[k]["iou"].append(intersection / union if union else float("nan"))
            per_k[k]["overlap"].append(intersection / m if m else float("nan"))

    return {
        k: {
            "iou": np.asarray(per_k[k]["iou"], dtype=np.float64),
            "overlap": np.asarray(per_k[k]["overlap"], dtype=np.float64),
        }
        for k in ks
    }


def evaluate_frame(
    frame: pd.DataFrame,
    signature_cols: list[str],
    align_index: np.ndarray,
    file_to_submitter: dict[str, str],
    labels_by_submitter: dict[str, np.ndarray],
    ks: Sequence[int],
) -> tuple[
    dict[int, dict[str, np.ndarray]], dict[str, int], np.ndarray, np.ndarray, np.ndarray
]:
    """Model top-k metrics for one frame, plus the gathered ``trues`` / ``preds`` / ``n_active``.

    The trailing arrays are returned so the caller can pool them across folds
    and score the baselines + distribution/AUROC summary on the identical slide
    set.
    """
    trues, preds, n_active, counts = _gather_evaluable(
        frame, signature_cols, align_index, file_to_submitter, labels_by_submitter
    )
    arrays = _topk_metrics(trues, preds, n_active, ks, len(signature_cols))
    return arrays, counts, trues, preds, n_active


def _hypergeom_topk_expectation(m: int, num_signatures: int) -> tuple[float, float]:
    """Exact ``E[IoU]``, ``E[overlap@k]`` for a random m-subset vs a fixed m-target.

    Picking ``m`` of ``S`` signatures uniformly at random, the intersection with
    a fixed ``m``-element true set is hypergeometric, so ``E[overlap] = m/S`` and
    ``E[IoU] = Σ P(i) · i/(2m − i)``. This is the no-information floor.
    """
    if m <= 0:
        return float("nan"), float("nan")
    if m >= num_signatures:
        return 1.0, 1.0
    denominator = math.comb(num_signatures, m)
    expected_iou = 0.0
    expected_overlap = 0.0
    for i in range(0, m + 1):
        if m - i > num_signatures - m:  # outside the hypergeometric support
            continue
        probability = (
            math.comb(m, i) * math.comb(num_signatures - m, m - i) / denominator
        )
        expected_overlap += probability * (i / m)
        expected_iou += probability * (i / (2 * m - i))
    return expected_iou, expected_overlap


def random_baseline(
    n_active: np.ndarray, ks: Sequence[int], num_signatures: int
) -> dict[int, dict[str, np.ndarray]]:
    """Per-slide expected IoU / overlap@k for a uniformly-random predictor."""
    cache: dict[int, tuple[float, float]] = {}
    per_k: dict[int, dict[str, list[float]]] = {
        k: {"iou": [], "overlap": []} for k in ks
    }
    for active in n_active:
        for k in ks:
            m = min(k, int(active), num_signatures)
            if m not in cache:
                cache[m] = _hypergeom_topk_expectation(m, num_signatures)
            iou, overlap = cache[m]
            per_k[k]["iou"].append(iou)
            per_k[k]["overlap"].append(overlap)
    return {
        k: {
            "iou": np.asarray(per_k[k]["iou"], dtype=np.float64),
            "overlap": np.asarray(per_k[k]["overlap"], dtype=np.float64),
        }
        for k in ks
    }


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax -> predicted exposure distribution (each row sums to 1).

    The subtask head trains with KL(target || softmax(prediction)), so the raw
    predicted columns are logits; softmax recovers the predicted exposures that
    live on the same simplex as the labels.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def total_variation(dist_a: np.ndarray, dist_b: np.ndarray) -> np.ndarray:
    """Per-row total-variation distance ``0.5 * |a - b|_1`` (rows are distributions).

    0 = identical profiles, 1 = disjoint support. Interpretable as the fraction
    of probability mass the prediction misplaces.
    """
    return 0.5 * np.abs(dist_a - dist_b).sum(axis=1)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Renormalize each row to sum to 1 (no-op when already on the simplex)."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return matrix / row_sums


def _average_ranks(sorted_values: np.ndarray) -> np.ndarray:
    """1-based average ranks for an already-sorted 1-D array (ties share rank)."""
    n = sorted_values.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """One-vs-rest AUROC via the rank statistic; NaN if a class is degenerate.

    Identical to the implementation in ``subtyping_auroc.py`` (Mann-Whitney U
    with average-rank tie handling), kept local so this script needs no extra
    imports.
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = _average_ranks(scores[order])
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auroc_active(
    trues: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[int, float], int]:
    """Macro AUROC for 'signature active (true > 0) vs not', over scorable signatures.

    Per signature column: positives are slides where the signature is truly
    active, the score is the predicted exposure. Signatures that are active (or
    inactive) in every evaluated slide are degenerate and excluded from the
    macro mean. Because the score ranks *patients*, a marginal predictor that
    assigns every patient the same exposure scores exactly 0.5.
    """
    per_signature: dict[int, float] = {}
    valid: list[float] = []
    for j in range(trues.shape[1]):
        auroc = _binary_auroc(trues[:, j] > 0.0, scores[:, j])
        per_signature[j] = auroc
        if not np.isnan(auroc):
            valid.append(auroc)
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_signature, len(valid)


def distribution_summary(
    predicted_dist: np.ndarray,
    trues: np.ndarray,
    signature_cols: list[str],
    *,
    keep_per_signature: bool = True,
) -> dict:
    """k-independent metrics for an already-normalized predicted distribution.

    ``predicted_dist`` is an ``(N, S)`` row-stochastic matrix -- ``softmax`` of
    the model logits, or a baseline distribution broadcast across the N slides.
    Returns mean total variation against the true profiles plus the
    per-signature "active vs not" macro AUROC scored by ``predicted_dist``.
    Used identically for the model and the prevalence / random baselines.
    """
    true_dist = _normalize_rows(trues)
    tv = total_variation(predicted_dist, true_dist)
    macro_auroc, per_signature_idx, n_scored = macro_auroc_active(trues, predicted_dist)
    summary = {
        "total_variation": _nanmean(tv),
        "macro_auroc_active": macro_auroc,
        "n_auroc_signatures": n_scored,
        "n_signatures": int(trues.shape[1]),
    }
    if keep_per_signature:
        active_support = (trues > 0.0).sum(axis=0)
        summary["per_signature_auroc"] = {
            signature_cols[j]: (None if np.isnan(auroc) else float(auroc))
            for j, auroc in per_signature_idx.items()
        }
        summary["per_signature_support"] = {
            signature_cols[j]: int(active_support[j]) for j in range(trues.shape[1])
        }
    return summary


def summary_metrics(
    trues: np.ndarray, preds_logits: np.ndarray, signature_cols: list[str]
) -> dict:
    """Model TV + macro AUROC; softmaxes the logits onto the simplex first."""
    return distribution_summary(_softmax_rows(preds_logits), trues, signature_cols)


def label_activity(trues: np.ndarray, signature_cols: list[str]) -> dict:
    """Per-signature ground-truth activity over the evaluated slides.

    Each signature's ``mean_exposure`` is its average true contribution across
    slides (labels are per-slide distributions, so this is the cohort-mean
    profile) and ``support`` is the number of slides where it is active (> 0).
    Used to report which signatures the cohort actually carries, for comparison
    against which the model predicts best.
    """
    mean_exposure = trues.mean(axis=0)
    support = (trues > 0.0).sum(axis=0)
    return {
        "per_signature": {
            signature_cols[j]: {
                "mean_exposure": float(mean_exposure[j]),
                "support": int(support[j]),
            }
            for j in range(trues.shape[1])
        }
    }


def compute_baselines(
    trues: np.ndarray,
    n_active: np.ndarray,
    labels_by_submitter: dict[str, np.ndarray],
    align_index: np.ndarray,
    signature_cols: list[str],
    ks: Sequence[int],
) -> dict:
    """Prevalence + random baselines on the pooled evaluated slides.

    * prevalence: ignore the image and predict the cohort-mean exposure profile
      (the same vector for every slide), scored against each slide's true
      top-m. This is the bar a real model must beat -- BRCA signature profiles
      are concentrated, so the marginal already recovers many top signatures.
    * random: the exact expected IoU / overlap@k of a uniformly-random predictor
      (the no-information floor, ``E[overlap] = m/S``).

    The cohort-mean is taken over every submitter in the labels table (the
    population marginal), independent of which slides are in the eval set.
    """
    num_signatures = len(signature_cols)

    label_matrix = np.stack(list(labels_by_submitter.values()))
    prevalence = label_matrix.mean(axis=0)[align_index]
    prevalence_preds = np.broadcast_to(prevalence, trues.shape)
    prevalence_arrays = _topk_metrics(
        trues, prevalence_preds, n_active, ks, num_signatures
    )
    random_arrays = random_baseline(n_active, ks, num_signatures)

    # k-independent distribution / detection metrics on the same pooled slides.
    # Prevalence "predicts" the cohort-mean profile for every slide; random
    # "predicts" the uniform (max-entropy, no-information) distribution. Both are
    # broadcast to (N, S) and scored exactly like the model. Their macro AUROC is
    # 0.5 by construction (a constant per-signature score ranks no patient above
    # another), which is precisely the chance line the model must beat.
    prevalence_dist = np.broadcast_to(
        _normalize_rows(prevalence[np.newaxis, :]), trues.shape
    )
    prevalence_summary = distribution_summary(
        prevalence_dist, trues, signature_cols, keep_per_signature=False
    )
    uniform_dist = np.broadcast_to(
        np.full((1, num_signatures), 1.0 / num_signatures), trues.shape
    )
    random_summary = distribution_summary(
        uniform_dist, trues, signature_cols, keep_per_signature=False
    )

    ranking = np.argsort(-prevalence, kind="mergesort")
    top_prevalent = [signature_cols[i] for i in ranking[: max(ks)]]

    return {
        "prevalence_source": (
            "cohort-mean exposure over all submitters in the labels table"
        ),
        "random_source": "uniform (max-entropy) distribution over all signatures",
        "prevalence_top_signatures": top_prevalent,
        "prevalence_summary": prevalence_summary,
        "random_summary": random_summary,
        "prevalence": {
            str(k): {
                "overlap": _nanmean(prevalence_arrays[k]["overlap"]),
                "iou": _nanmean(prevalence_arrays[k]["iou"]),
            }
            for k in ks
        },
        "random": {
            str(k): {
                "overlap": _nanmean(random_arrays[k]["overlap"]),
                "iou": _nanmean(random_arrays[k]["iou"]),
            }
            for k in ks
        },
    }


def _nanmean(values: np.ndarray) -> float:
    """Mean ignoring NaNs; NaN if empty."""
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> dict | None:
    """Bootstrap std + 95% percentile CI of the mean by resampling slides."""
    clean = values[~np.isnan(values)]
    if n_boot <= 0 or clean.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = clean[rng.integers(0, clean.size, size=clean.size)]
        means[b] = sample.mean()
    return {
        "std": float(means.std(ddof=1)) if means.size > 1 else 0.0,
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def build_report(
    fold_results: list[tuple[str, dict[int, dict[str, np.ndarray]], dict[str, int]]],
    ks: Sequence[int],
    n_boot: int,
    seed: int,
) -> dict:
    """Pool slides across folds and assemble the metrics dict."""
    report: dict = {
        "n_folds": len(fold_results),
        "top_k": list(ks),
        "counts": {
            "evaluated": sum(counts["evaluated"] for _, _, counts in fold_results),
            "missing_submitter": sum(
                counts["missing_submitter"] for _, _, counts in fold_results
            ),
            "missing_label": sum(
                counts["missing_label"] for _, _, counts in fold_results
            ),
            "no_active_signatures": sum(
                counts["no_active_signatures"] for _, _, counts in fold_results
            ),
        },
        "per_fold": [],
        "metrics": {},
    }
    total_evaluated = report["counts"]["evaluated"]
    total_active = sum(counts["active_total"] for _, _, counts in fold_results)
    report["mean_active_signatures"] = (
        total_active / total_evaluated if total_evaluated else float("nan")
    )

    for path, arrays, counts in fold_results:
        report["per_fold"].append(
            {
                "path": path,
                "n_evaluated": counts["evaluated"],
                "iou": {str(k): _nanmean(arrays[k]["iou"]) for k in ks},
                "overlap": {str(k): _nanmean(arrays[k]["overlap"]) for k in ks},
            }
        )

    for k in ks:
        pooled_iou = np.concatenate([arrays[k]["iou"] for _, arrays, _ in fold_results])
        pooled_overlap = np.concatenate(
            [arrays[k]["overlap"] for _, arrays, _ in fold_results]
        )
        fold_iou_means = np.asarray(
            [_nanmean(arrays[k]["iou"]) for _, arrays, _ in fold_results]
        )
        fold_overlap_means = np.asarray(
            [_nanmean(arrays[k]["overlap"]) for _, arrays, _ in fold_results]
        )

        entry: dict = {
            "pooled_iou": _nanmean(pooled_iou),
            "pooled_overlap": _nanmean(pooled_overlap),
            "macro_fold_iou": _nanmean(fold_iou_means),
            "macro_fold_overlap": _nanmean(fold_overlap_means),
            "macro_fold_iou_std": (
                float(np.nanstd(fold_iou_means, ddof=1))
                if np.sum(~np.isnan(fold_iou_means)) > 1
                else 0.0
            ),
            "macro_fold_overlap_std": (
                float(np.nanstd(fold_overlap_means, ddof=1))
                if np.sum(~np.isnan(fold_overlap_means)) > 1
                else 0.0
            ),
        }
        iou_ci = _bootstrap_ci(pooled_iou, n_boot, seed)
        overlap_ci = _bootstrap_ci(pooled_overlap, n_boot, seed)
        if iou_ci is not None:
            entry["pooled_iou_std"] = iou_ci["std"]
            entry["pooled_iou_ci95"] = iou_ci["ci95"]
        if overlap_ci is not None:
            entry["pooled_overlap_std"] = overlap_ci["std"]
            entry["pooled_overlap_ci95"] = overlap_ci["ci95"]
        report["metrics"][str(k)] = entry

    if n_boot > 0:
        report["n_bootstrap"] = n_boot
    return report


def print_report(
    report: dict, subtask: str, top_auroc: int = 10, top_active: int = 10
) -> None:
    """Human-readable table to stdout."""
    counts = report["counts"]
    print(f"\nsubtask: {subtask}   folds: {report['n_folds']}")
    print(
        f"slides evaluated: {counts['evaluated']}   "
        f"skipped (no submitter): {counts['missing_submitter']}   "
        f"skipped (no label): {counts['missing_label']}   "
        f"skipped (no active signature): {counts['no_active_signatures']}"
    )
    print(
        f"mean active signatures / slide: {report['mean_active_signatures']:.2f}   "
        "(top-k is capped to this per slide)\n"
    )

    has_ci = "n_bootstrap" in report
    header = (
        f"{'k':>4}{'IoU (pooled)':>22}{'overlap@k (pooled)':>22}{'IoU (fold avg)':>22}"
    )
    print(header)
    print("-" * len(header))
    for k in report["top_k"]:
        entry = report["metrics"][str(k)]
        if has_ci and "pooled_iou_ci95" in entry:
            lo, hi = entry["pooled_iou_ci95"]
            iou_cell = f"{entry['pooled_iou']:.4f} ±{entry['pooled_iou_std']:.4f}"
            overlap_cell = (
                f"{entry['pooled_overlap']:.4f} ±{entry['pooled_overlap_std']:.4f}"
            )
        else:
            iou_cell = f"{entry['pooled_iou']:.4f}"
            overlap_cell = f"{entry['pooled_overlap']:.4f}"
        fold_cell = f"{entry['macro_fold_iou']:.4f} ±{entry['macro_fold_iou_std']:.4f}"
        print(f"{k:>4}{iou_cell:>22}{overlap_cell:>22}{fold_cell:>22}")
    print("-" * len(header))
    if has_ci:
        print(
            f"(pooled ±std and the fold-avg are over {report['n_folds']} fold(s); "
            f"±std on pooled is the bootstrap SE, n={report['n_bootstrap']})"
        )
    print()

    summary = report.get("summary")
    if summary is not None:
        baselines = report.get("baselines")
        prevalence_summary = (
            baselines.get("prevalence_summary") if baselines is not None else None
        )
        random_summary = (
            baselines.get("random_summary") if baselines is not None else None
        )

        def _cell(value: float | None) -> str:
            return "n/a" if value is None or np.isnan(value) else f"{value:.4f}"

        print("distributional & detection (pooled over evaluated slides):")
        tv_line = (
            "  total variation (lower=better):        "
            f"model {_cell(summary['total_variation'])}"
        )
        if prevalence_summary is not None:
            tv_line += f"   prevalence {_cell(prevalence_summary['total_variation'])}"
        if random_summary is not None:
            tv_line += f"   random {_cell(random_summary['total_variation'])}"
        print(tv_line)

        auroc_line = (
            "  macro AUROC active-vs-not (0.5=chance): "
            f"model {_cell(summary['macro_auroc_active'])}"
        )
        if prevalence_summary is not None:
            auroc_line += (
                f"   prevalence {_cell(prevalence_summary['macro_auroc_active'])}"
            )
        if random_summary is not None:
            auroc_line += f"   random {_cell(random_summary['macro_auroc_active'])}"
        print(auroc_line)
        print(
            f"  (macro AUROC over {summary['n_auroc_signatures']}/"
            f"{summary['n_signatures']} signatures with both classes present; "
            "prevalence/random = 0.5 by construction)"
        )
        print()

        # Most discriminable signatures by per-signature AUROC. Support (number
        # of active patients) is shown because a high AUROC on tiny support is
        # noise, not signal.
        per_signature = summary.get("per_signature_auroc") or {}
        support = summary.get("per_signature_support") or {}
        ranked = sorted(
            (
                (name, auroc)
                for name, auroc in per_signature.items()
                if auroc is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_auroc > 0 and ranked:
            shown = ranked[:top_auroc]
            print(f"top {len(shown)} signatures by AUROC (active-vs-not):")
            print(f"  {'signature':<12}{'AUROC':>8}{'active slides':>16}")
            for name, auroc in shown:
                print(f"  {name:<12}{auroc:>8.4f}{support.get(name, 0):>16}")
            print()

    label = report.get("label_activity")
    if top_active > 0 and label:
        ranked_active = sorted(
            label["per_signature"].items(),
            key=lambda item: item[1]["mean_exposure"],
            reverse=True,
        )[:top_active]
        if ranked_active:
            print(
                f"top {len(ranked_active)} most active signatures "
                "(label, by mean exposure over evaluated slides):"
            )
            print(f"  {'signature':<12}{'mean exposure':>16}{'active slides':>16}")
            for name, stats in ranked_active:
                print(
                    f"  {name:<12}{stats['mean_exposure']:>16.4f}"
                    f"{stats['support']:>16}"
                )
            print()

    if "baselines" in report:
        prevalence = report["baselines"]["prevalence"]
        random_b = report["baselines"]["random"]
        bheader = (
            f"{'k':>4}{'model O / IoU':>22}{'prevalence O / IoU':>22}"
            f"{'random O / IoU':>22}{'lift(O)':>10}"
        )
        print("baselines (pooled over evaluated slides; O = overlap@k):")
        print(bheader)
        print("-" * len(bheader))
        for k in report["top_k"]:
            entry = report["metrics"][str(k)]
            prev = prevalence[str(k)]
            rand = random_b[str(k)]
            model_cell = f"{entry['pooled_overlap']:.4f} / {entry['pooled_iou']:.4f}"
            prev_cell = f"{prev['overlap']:.4f} / {prev['iou']:.4f}"
            rand_cell = f"{rand['overlap']:.4f} / {rand['iou']:.4f}"
            lift = entry["pooled_overlap"] - prev["overlap"]
            print(f"{k:>4}{model_cell:>22}{prev_cell:>22}{rand_cell:>22}{lift:>+10.4f}")
        print("-" * len(bheader))
        print(
            "lift = model overlap@k − prevalence overlap@k  "
            "(>0 ⇒ beats predicting the average profile)"
        )
        print(
            "prevalence predicts: "
            f"{', '.join(report['baselines']['prevalence_top_signatures'])}\n"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Top-k signature-set agreement (IoU + overlap@k) for slide-level "
            "mutational-signature regression predictions."
        ),
    )
    parser.add_argument(
        "pred_paths",
        nargs="+",
        help="One or more <subtask>.txt prediction TSVs (pooled if multiple folds).",
    )
    parser.add_argument(
        "--root-dir",
        default="data/TCGA-BRCA",
        help=(
            "Dataset root holding atlases/ (slide_subtask_atlas.txt + "
            "manifest_atlas.txt) used to resolve the labels and the "
            "file_id->submitter_id map. Use data/TCGA-BRCA-test for the test set."
        ),
    )
    parser.add_argument(
        "--subtask",
        default=None,
        help="Subtask / atlas key (e.g. sbs_regression). Inferred from the file stem if omitted.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Explicit contribution TSV path (overrides slide_subtask_atlas.txt).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Explicit GDC manifest path (overrides manifest_atlas.txt).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5, 10],
        metavar="K",
        help="One or more k values for the top-k signature sets (default: 5 10).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for ±std / 95%% CI on the pooled means (0 = off).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--top-auroc",
        type=int,
        default=5,
        metavar="N",
        help=(
            "List the N signatures with the highest per-signature AUROC "
            "(active-vs-not), with their support. 0 disables the listing."
        ),
    )
    parser.add_argument(
        "--top-active",
        type=int,
        default=0,
        metavar="N",
        help=(
            "List the N most active signatures in the label (by mean true "
            "exposure over evaluated slides), with their support. 0 disables it."
        ),
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help=(
            "Also score a prevalence predictor (always predict the cohort-mean "
            "profile) and a uniformly-random predictor on the same slides, and "
            "report the model's lift over prevalence."
        ),
    )
    parser.add_argument(
        "--save", default=None, help="Optional path to write the metrics as JSON."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    ks = sorted({int(k) for k in args.top_k if int(k) > 0})
    if not ks:
        raise ValueError("--top-k must contain at least one positive integer.")

    subtask = infer_subtask(args.pred_paths, args.subtask)
    frames = load_prediction_frames(args.pred_paths)
    signature_cols = signature_columns(frames[0][1])

    labels_path = resolve_labels_path(args.root_dir, subtask, args.labels)
    labels_by_submitter, signature_names = load_signature_labels(labels_path)
    manifest_path = resolve_manifest_path(args.root_dir, args.manifest)
    file_to_submitter = load_file_to_submitter(manifest_path)

    # Align the label vector (in signature_names order) to the prediction
    # columns (in file order), by name, so the two top-k sets index the same
    # signatures even if the column orders ever diverge.
    name_to_label_index = {name: idx for idx, name in enumerate(signature_names)}
    missing = [c for c in signature_cols if c not in name_to_label_index]
    if missing:
        raise ValueError(
            f"Prediction signature column(s) absent from the labels file "
            f"{labels_path}: {missing}. Are predictions and labels the same "
            f"subtask ({subtask})?"
        )
    align_index = np.asarray(
        [name_to_label_index[c] for c in signature_cols], dtype=np.int64
    )

    print(f"labels:   {labels_path}")
    print(f"manifest: {manifest_path}")
    print(f"signatures: {len(signature_cols)}   top-k: {ks}")

    fold_results: list[tuple[str, dict, dict]] = []
    pooled_trues: list[np.ndarray] = []
    pooled_preds: list[np.ndarray] = []
    pooled_active: list[np.ndarray] = []
    for path, frame in frames:
        arrays, counts, trues, preds, n_active = evaluate_frame(
            frame,
            signature_cols,
            align_index,
            file_to_submitter,
            labels_by_submitter,
            ks,
        )
        fold_results.append((path, arrays, counts))
        pooled_trues.append(trues)
        pooled_preds.append(preds)
        pooled_active.append(n_active)

    report = build_report(fold_results, ks, args.bootstrap, args.seed)
    report["subtask"] = subtask
    report["labels_path"] = labels_path
    report["manifest_path"] = manifest_path
    report["signatures"] = signature_cols

    if report["counts"]["evaluated"] == 0:
        raise RuntimeError(
            "No slides could be evaluated -- none of the prediction file_ids "
            "mapped to a submitter with a signature label. Check --root-dir / "
            "--manifest / --labels match the predicted dataset."
        )

    trues_all = np.concatenate(pooled_trues, axis=0)
    preds_all = np.concatenate(pooled_preds, axis=0)
    active_all = np.concatenate(pooled_active, axis=0)

    report["summary"] = summary_metrics(trues_all, preds_all, signature_cols)
    report["label_activity"] = label_activity(trues_all, signature_cols)
    if args.baselines:
        report["baselines"] = compute_baselines(
            trues_all,
            active_all,
            labels_by_submitter,
            align_index,
            signature_cols,
            ks,
        )

    print_report(report, subtask, top_auroc=args.top_auroc, top_active=args.top_active)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
