#!/usr/bin/env python3
"""Evaluate VGGT relative camera pose on the Pi3 TD-HOMES benchmark maps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from evaluation.td_homes import (
    DATASET_NAMES,
    EXPECTED_ALL_OVERLAP_COUNTS,
    OVERLAP_NAMES,
    TD_REFERENCE_FILENAMES,
    compute_pose_metrics,
    iter_map_entries,
    json_ready,
    load_sample,
    relative_pose_errors,
    safe_sample_stem,
    write_one_row_csv,
)


DEFAULT_PI3_ROOT = Path("/home/ma-user/Projects/Pi3_early_fusion/Pi3_TD_Fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VGGT using Pi3_early_fusion's fixed TD-HOMES sequence maps. "
            "By default the TD view is model input/reference index 0; --no-td "
            "runs the same mapped perspective views without it."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pi3-root", type=Path, default=DEFAULT_PI3_ROOT)
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/relpose-angular_VGGT"))
    parser.add_argument("--model", default="facebook/VGGT-1B", help="Checkpoint file or Hugging Face model id")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--overlaps", nargs="+", choices=OVERLAP_NAMES, default=list(OVERLAP_NAMES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument(
        "--no-td",
        action="store_true",
        help="Run VGGT on perspective views only; do not include the TD image",
    )
    parser.add_argument("--max-sequences", type=int, default=None, help="Limit each dataset/overlap for smoke tests")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-aggregate", action="store_true", help="Leave final aggregation to a coordinator")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate existing complete sample files without loading VGGT")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate every selected path without loading VGGT")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.map_dir is None:
        args.map_dir = args.pi3_root / "datasets" / "seq-id-maps"
    if not args.data_root.is_dir():
        reference_data = args.pi3_root.parent / "data"
        if reference_data.is_dir():
            args.data_root = reference_data
    args.data_root = args.data_root.resolve()
    args.map_dir = args.map_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    if not args.map_dir.is_dir():
        raise FileNotFoundError(f"Pi3 sequence-map directory does not exist: {args.map_dir}")


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_model(model_source: str, device: torch.device) -> VGGT:
    source_path = Path(model_source).expanduser()
    if source_path.is_file():
        model = VGGT()
        if source_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(source_path), device="cpu")
        else:
            try:
                checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(source_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint)
    else:
        model = VGGT.from_pretrained(model_source)
    return model.eval().to(device)


def infer_perspective_extrinsics(
    model: VGGT,
    image_paths: tuple[Path, ...],
    device: torch.device,
    dtype: torch.dtype,
    preprocess_mode: str,
    include_td: bool = True,
) -> tuple[torch.Tensor, tuple[int, int]]:
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=preprocess_mode
    ).to(device)
    minimum = 2 if include_td else 1
    if len(images) < minimum:
        mode = "one TD reference plus a perspective view" if include_td else "a perspective view"
        raise ValueError(f"Expected at least {mode}")
    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        predictions = model(images)
    with torch.inference_mode():
        extrinsics, _ = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), images.shape[-2:]
        )
    # With TD, index zero is the uncalibrated reference and is excluded. Without
    # TD, every model input is a calibrated perspective view and is retained.
    perspective = extrinsics[0, 1:] if include_td else extrinsics[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(
        perspective.shape[0], 1, 4
    )
    return torch.cat([perspective, bottom], dim=1).double(), tuple(images.shape[-2:])


def sample_result_path(output_dir: Path, dataset: str, overlap: str, sequence_name: str) -> Path:
    return output_dir / "samples" / dataset / f"{safe_sample_stem(overlap, sequence_name)}.npz"


def save_sample_result(
    path: Path,
    sample,
    rotation_error: np.ndarray,
    translation_error: np.ndarray,
    predicted_extrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    seconds: float,
    td_included: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dataset=sample.dataset,
        overlap=sample.overlap,
        sequence_name=sample.sequence_name,
        frame_ids=np.asarray(sample.frame_ids),
        td_included=np.asarray(td_included),
        td_reference_path=str(sample.td_path) if td_included else "",
        td_reference_index=np.asarray(0 if td_included else -1),
        perspective_image_paths=np.asarray([str(path) for path in sample.image_paths]),
        rotation_error=rotation_error,
        translation_error=translation_error,
        predicted_w2c=predicted_extrinsics.detach().cpu().numpy(),
        ground_truth_w2c=sample.gt_extrinsics,
        input_height=np.asarray(image_hw[0]),
        input_width=np.asarray(image_hw[1]),
        inference_seconds=np.asarray(seconds),
    )


def write_aggregates(output_dir: Path, datasets: list[str], partial: bool) -> None:
    populations = {}
    for dataset in datasets:
        sample_files = sorted((output_dir / "samples" / dataset).glob("*.npz"))
        if not sample_files:
            continue
        rotations, translations, rows = [], [], []
        for sample_file in sample_files:
            with np.load(sample_file, allow_pickle=False) as result:
                r_error = result["rotation_error"].astype(np.float64)
                t_error = result["translation_error"].astype(np.float64)
                rotations.append(r_error)
                translations.append(t_error)
                rows.append(
                    {
                        "dataset": str(result["dataset"]),
                        "overlap": str(result["overlap"]),
                        "sequence_name": str(result["sequence_name"]),
                        "num_views": int(len(result["frame_ids"])),
                        "num_pairs": int(len(r_error)),
                        "MeanRE": float(np.mean(r_error)) if len(r_error) else float("nan"),
                        "MeanTE": float(np.mean(t_error)) if len(t_error) else float("nan"),
                        "MRE": float(np.median(r_error)) if len(r_error) else float("nan"),
                        "MTE": float(np.median(t_error)) if len(t_error) else float("nan"),
                        "inference_seconds": float(result["inference_seconds"]),
                        "td_reference_index": int(result["td_reference_index"]),
                        "td_reference_path": str(result["td_reference_path"]),
                    }
                )
        rotation = np.concatenate(rotations)
        translation = np.concatenate(translations)
        populations[dataset] = (rotation, translation)
        metrics = {
            "dataset": dataset,
            "partial": partial,
            "num_samples": len(sample_files),
            "num_pairs": len(rotation),
            **compute_pose_metrics(rotation, translation),
        }
        write_one_row_csv(output_dir / f"{dataset}-metric.csv", metrics)
        np.savez_compressed(
            output_dir / f"{dataset}-errors.npz",
            rotation_error=rotation,
            translation_error=translation,
        )
        per_sequence_path = output_dir / dataset / "per-sequence.csv"
        per_sequence_path.parent.mkdir(parents=True, exist_ok=True)
        with per_sequence_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    homes_members = [populations[name] for name in ("TD_HM3D", "FP_Stru3D") if name in populations]
    if len(homes_members) == 2:
        rotation = np.concatenate([value[0] for value in homes_members])
        translation = np.concatenate([value[1] for value in homes_members])
        metrics = {
            "dataset": "TD_HOMES",
            "partial": partial,
            "num_samples": sum(
                len(list((output_dir / "samples" / name).glob("*.npz")))
                for name in ("TD_HM3D", "FP_Stru3D")
            ),
            "num_pairs": len(rotation),
            **compute_pose_metrics(rotation, translation),
        }
        write_one_row_csv(output_dir / "TD_HOMES-metric.csv", metrics)
        np.savez_compressed(
            output_dir / "TD_HOMES-errors.npz",
            rotation_error=rotation,
            translation_error=translation,
        )


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    counts = {dataset: 0 for dataset in args.datasets}
    overlap_counts = {(dataset, overlap): 0 for dataset in args.datasets for overlap in args.overlaps}
    failures = []
    for dataset, overlap, sequence_name, frame_ids in iter_map_entries(
        args.map_dir, args.datasets, args.overlaps
    ):
        key = (dataset, overlap)
        if args.max_sequences is not None and overlap_counts[key] >= args.max_sequences:
            continue
        overlap_counts[key] += 1
        counts[dataset] += 1
        if args.aggregate_only:
            continue
        try:
            sample = load_sample(args.data_root, dataset, overlap, sequence_name, frame_ids)
            if not args.no_td and sample.model_image_paths[0] != sample.td_path:
                raise AssertionError("TD reference is not model input index zero")
            selected.append(sample)
        except Exception as exc:
            failures.append(
                {"dataset": dataset, "overlap": overlap, "sequence_name": sequence_name, "error": str(exc)}
            )

    partial = args.max_sequences is not None or tuple(args.overlaps) != OVERLAP_NAMES
    manifest = {
        "model": args.model,
        "data_root": args.data_root,
        "map_dir": args.map_dir,
        "output_dir": args.output_dir,
        "datasets": args.datasets,
        "overlaps": args.overlaps,
        "counts": counts,
        "expected_all_overlap_counts": EXPECTED_ALL_OVERLAP_COUNTS,
        "partial": partial,
        "td_included": not args.no_td,
        "td_reference_index": 0 if not args.no_td else None,
        "td_scored": False,
        "td_reference_filenames": TD_REFERENCE_FILENAMES if not args.no_td else None,
        "failed_path_validations": failures,
        "num_shards": args.num_shards,
    }
    # Only the coordinator owns shared metadata during multi-process runs.
    if args.shard_index == 0:
        with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=json_ready)
            handle.write("\n")
    if failures:
        preview = "\n".join(
            f"  {row['dataset']}/{row['overlap']}/{row['sequence_name']}: {row['error']}"
            for row in failures[:10]
        )
        raise RuntimeError(f"{len(failures)} samples failed path/pose validation:\n{preview}")
    if not partial:
        mismatches = {
            dataset: (counts[dataset], EXPECTED_ALL_OVERLAP_COUNTS[dataset])
            for dataset in args.datasets
            if counts[dataset] != EXPECTED_ALL_OVERLAP_COUNTS[dataset]
        }
        if mismatches:
            raise RuntimeError(f"Sequence-map counts do not match Pi3 benchmark: {mismatches}")
    input_mode = "TD reference at index 0" if not args.no_td else "perspective views only (no TD)"
    status = "Mapped" if args.aggregate_only else "Validated"
    selected_count = sum(counts.values()) if args.aggregate_only else len(selected)
    print(f"{status} {selected_count} samples; input mode: {input_mode}.")
    if args.dry_run:
        print(f"Dry-run manifest: {args.output_dir / 'manifest.json'}")
        return
    if args.aggregate_only:
        missing = {}
        for dataset in args.datasets:
            actual = len(list((args.output_dir / "samples" / dataset).glob("*.npz")))
            if actual != counts[dataset]:
                missing[dataset] = {"expected": counts[dataset], "actual": actual}
        if missing:
            raise RuntimeError(f"Cannot aggregate an incomplete evaluation: {missing}")
        write_aggregates(args.output_dir, args.datasets, partial)
        print(f"Aggregated complete results: {args.output_dir}")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = select_dtype(args.dtype, device)
    model = load_model(args.model, device)
    completed = 0
    shard_samples = selected[args.shard_index :: args.num_shards]
    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: "
        f"{len(shard_samples)} of {len(selected)} samples"
    )
    for index, sample in enumerate(shard_samples, start=1):
        result_path = sample_result_path(
            args.output_dir, sample.dataset, sample.overlap, sample.sequence_name
        )
        if result_path.is_file() and not args.overwrite:
            completed += 1
            print(f"[{index}/{len(shard_samples)}] resume {sample.dataset}/{sample.overlap}/{sample.sequence_name}")
            continue
        start = time.perf_counter()
        predicted, image_hw = infer_perspective_extrinsics(
            model,
            sample.image_paths if args.no_td else sample.model_image_paths,
            device,
            dtype,
            args.preprocess_mode,
            include_td=not args.no_td,
        )
        seconds = time.perf_counter() - start
        ground_truth = torch.from_numpy(sample.gt_extrinsics).to(device=device, dtype=torch.float64)
        rotation_error, translation_error = relative_pose_errors(predicted, ground_truth)
        save_sample_result(
            result_path,
            sample,
            rotation_error,
            translation_error,
            predicted,
            image_hw,
            seconds,
            td_included=not args.no_td,
        )
        completed += 1
        print(
            f"[{index}/{len(shard_samples)}] {sample.dataset}/{sample.overlap}/{sample.sequence_name} "
            f"RE={rotation_error.mean() if len(rotation_error) else float('nan'):.2f} "
            f"TE={translation_error.mean() if len(translation_error) else float('nan'):.2f} {seconds:.2f}s",
            flush=True,
        )
    if not args.skip_aggregate:
        write_aggregates(args.output_dir, args.datasets, partial)
    print(f"Completed {completed} samples. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
