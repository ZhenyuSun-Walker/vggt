#!/usr/bin/env python3
"""Evaluate VGGT multi-view reconstruction with Pi3 TD-HOMES settings."""

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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_td_homes import DEFAULT_PI3_ROOT, load_model, resolve_paths, select_dtype
from evaluation.mv_recon_td_homes import (
    METRIC_NAMES,
    align_and_score_reconstruction,
    load_pi3_style_vggt_input,
    load_reconstruction_ground_truth,
    save_image_grid,
)
from evaluation.td_homes import (
    DATASET_NAMES,
    EXPECTED_ALL_OVERLAP_COUNTS,
    OVERLAP_NAMES,
    TD_REFERENCE_FILENAMES,
    iter_map_entries,
    json_ready,
    load_sample,
    safe_sample_stem,
    write_one_row_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pi3-root", type=Path, default=DEFAULT_PI3_ROOT)
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/mv-recon_VGGT"))
    parser.add_argument("--model", default="facebook/VGGT-1B")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--overlaps", nargs="+", choices=OVERLAP_NAMES, default=list(OVERLAP_NAMES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--load-img-size", type=int, default=512)
    parser.add_argument(
        "--no-td",
        action="store_true",
        help="Run VGGT on perspective views only; do not include the TD image",
    )
    parser.add_argument(
        "--metric-workers",
        type=int,
        default=-1,
        help="CPU workers per shard for each exact KD-tree query",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-save-pointclouds",
        action="store_true",
        help="Do not write Pi3-style pred/GT PLY files (metrics are unchanged)",
    )
    return parser.parse_args()


def infer_world_points(
    model, sample, device, dtype, load_img_size, target_hw, include_td=True
):
    images = load_pi3_style_vggt_input(
        sample, load_img_size, include_td=include_td
    ).to(device)
    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        prediction = model(images)
    # With TD, index zero is the reference and is excluded from GT scoring.
    # Without TD, every input/output corresponds to a perspective GT view.
    start_index = 1 if include_td else 0
    points = prediction["world_points"][0, start_index:].float()
    points = F.interpolate(
        points.permute(0, 3, 1, 2),
        target_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    return points.cpu().numpy(), tuple(images.shape[-2:])


def sample_json_path(output_dir, dataset, overlap, sequence_name):
    return output_dir / "samples" / dataset / f"{safe_sample_stem(overlap, sequence_name)}.json"


def write_sample_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_ready)
        handle.write("\n")


def aggregate(output_dir: Path, datasets: list[str], partial: bool) -> None:
    rows_by_dataset = {}
    for dataset in datasets:
        rows = []
        for path in sorted((output_dir / "samples" / dataset).glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                rows.append(json.load(handle))
        if not rows:
            continue
        rows_by_dataset[dataset] = rows
        output = output_dir / dataset / "_all_samples.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["dataset", "seq", *METRIC_NAMES, "num-valid-points", "Sim3-scale", "ICP-fitness", "ICP-inlier-rmse", "inference-seconds", "total-seconds", "td-reference-path", "td-reference-index"]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        metric = {
            "dataset": dataset,
            "partial": partial,
            "num_samples": len(rows),
            **{name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES},
        }
        write_one_row_csv(output_dir / f"{dataset}-metric.csv", metric)

    if "TD_HM3D" in rows_by_dataset and "FP_Stru3D" in rows_by_dataset:
        rows = rows_by_dataset["TD_HM3D"] + rows_by_dataset["FP_Stru3D"]
        output = output_dir / "TD_HOMES" / "_all_samples.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["dataset", "seq", *METRIC_NAMES]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        metric = {
            "dataset": "TD_HOMES",
            "partial": partial,
            "num_samples": len(rows),
            **{name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES},
        }
        write_one_row_csv(output_dir / "TD_HOMES-metric.csv", metric)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    if args.metric_workers == 0 or args.metric_workers < -1:
        raise ValueError("--metric-workers must be -1 or a positive integer")
    resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, failures = [], []
    mapped_index = 0
    counts = {dataset: 0 for dataset in args.datasets}
    overlap_counts = {(dataset, overlap): 0 for dataset in args.datasets for overlap in args.overlaps}
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
        current_index = mapped_index
        mapped_index += 1
        # Assign work before touching any dataset files.  This keeps resume
        # sharding stable while avoiding a full 1,629-sample validation pass in
        # every worker.  Completed samples have already passed validation and
        # can be skipped without reopening their source data.
        if current_index % args.num_shards != args.shard_index:
            continue
        result_path = sample_json_path(
            args.output_dir, dataset, overlap, sequence_name
        )
        if result_path.is_file() and not args.overwrite:
            continue
        if not args.dry_run:
            # Keep normal execution lazy: opening every RGB/pose/depth path up
            # front makes a resumed network-filesystem run wait several
            # minutes before it can do useful work.
            selected.append((dataset, overlap, sequence_name, tuple(frame_ids)))
            continue
        try:
            sample = load_sample(args.data_root, dataset, overlap, sequence_name, frame_ids)
            if not args.no_td and sample.model_image_paths[0] != sample.td_path:
                raise AssertionError("TD reference is not input index zero")
            selected.append(sample)
        except Exception as exc:
            failures.append({"dataset": dataset, "overlap": overlap, "sequence_name": sequence_name, "error": str(exc)})
    partial = args.max_sequences is not None or tuple(args.overlaps) != OVERLAP_NAMES
    manifest = {
        "task": "mv_recon",
        "model": args.model,
        "data_root": args.data_root,
        "map_dir": args.map_dir,
        "output_dir": args.output_dir,
        "datasets": args.datasets,
        "overlaps": args.overlaps,
        "counts": counts,
        "expected_all_overlap_counts": EXPECTED_ALL_OVERLAP_COUNTS,
        "partial": partial,
        "load_img_size": args.load_img_size,
        "td_included": not args.no_td,
        "td_reference_index": 0 if not args.no_td else None,
        "td_scored": False,
        "td_reference_filenames": TD_REFERENCE_FILENAMES if not args.no_td else None,
        "alignment": "Umeyama Sim(3) + point-to-point ICP (threshold=0.1m)",
        "metrics": list(METRIC_NAMES),
        "metric_workers_per_shard": args.metric_workers,
        "save_pointclouds": not args.no_save_pointclouds,
        "failed_path_validations": failures,
        "num_shards": args.num_shards,
    }
    if args.shard_index == 0:
        with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=json_ready)
            handle.write("\n")
    if failures:
        raise RuntimeError(f"{len(failures)} samples failed validation: {failures[:5]}")
    if not partial:
        mismatches = {
            dataset: (counts[dataset], EXPECTED_ALL_OVERLAP_COUNTS[dataset])
            for dataset in args.datasets
            if counts[dataset] != EXPECTED_ALL_OVERLAP_COUNTS[dataset]
        }
        if mismatches:
            raise RuntimeError(f"Pi3 map count mismatch: {mismatches}")
    input_mode = "TD reference at index 0" if not args.no_td else "perspective views only (no TD)"
    status = "Mapped" if args.aggregate_only else ("Validated" if args.dry_run else "Selected")
    selected_count = sum(counts.values()) if args.aggregate_only else len(selected)
    print(f"{status} {selected_count} mv_recon samples; input mode: {input_mode}.")
    if args.dry_run:
        return
    if args.aggregate_only:
        missing = {}
        for dataset in args.datasets:
            actual = len(list((args.output_dir / "samples" / dataset).glob("*.json")))
            if actual != counts[dataset]:
                missing[dataset] = {"expected": counts[dataset], "actual": actual}
        if missing:
            raise RuntimeError(f"Cannot aggregate incomplete mv_recon results: {missing}")
        aggregate(args.output_dir, args.datasets, partial)
        print(f"Aggregated complete mv_recon results: {args.output_dir}")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = select_dtype(args.dtype, device)
    model = load_model(args.model, device)
    shard_samples = selected
    print(f"Shard {args.shard_index + 1}/{args.num_shards}: {len(shard_samples)} samples")
    for index, item in enumerate(shard_samples, 1):
        dataset, overlap, sequence_name, frame_ids = item
        result_path = sample_json_path(
            args.output_dir, dataset, overlap, sequence_name
        )
        if result_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(shard_samples)}] resume {dataset}/{overlap}/{sequence_name}")
            continue
        try:
            sample = load_sample(
                args.data_root, dataset, overlap, sequence_name, frame_ids
            )
            if not args.no_td and sample.model_image_paths[0] != sample.td_path:
                raise AssertionError("TD reference is not input index zero")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {dataset}/{overlap}/{sequence_name}: {exc}"
            ) from exc
        started = time.perf_counter()
        ground_truth = load_reconstruction_ground_truth(sample, args.load_img_size)
        inference_started = time.perf_counter()
        predicted, input_hw = infer_world_points(
            model,
            sample,
            device,
            dtype,
            args.load_img_size,
            ground_truth.points.shape[1:3],
            include_td=not args.no_td,
        )
        inference_seconds = time.perf_counter() - inference_started
        logical_name = f"{sample.overlap}::{sample.sequence_name}"
        dataset_dir = args.output_dir / sample.dataset
        pred_ply = None if args.no_save_pointclouds else dataset_dir / f"{logical_name}-pred.ply"
        gt_ply = None if args.no_save_pointclouds else dataset_dir / f"{logical_name}-gt.ply"
        metrics = align_and_score_reconstruction(
            predicted,
            ground_truth,
            pred_ply=pred_ply,
            gt_ply=gt_ply,
            workers=args.metric_workers,
        )
        if not args.no_save_pointclouds:
            save_image_grid(ground_truth.images, dataset_dir / f"{logical_name}.png")
        payload = {
            "dataset": sample.dataset,
            "seq": logical_name,
            **metrics,
            "inference-seconds": inference_seconds,
            "total-seconds": time.perf_counter() - started,
            "input-height": input_hw[0],
            "input-width": input_hw[1],
            "td-included": not args.no_td,
            "td-reference-path": str(sample.td_path) if not args.no_td else "",
            "td-reference-index": 0 if not args.no_td else -1,
            "num-perspective-views": len(sample.image_paths),
        }
        write_sample_json(result_path, payload)
        print(
            f"[{index}/{len(shard_samples)}] {logical_name} "
            f"Acc={metrics['Acc-mean']:.4f} Comp={metrics['Comp-mean']:.4f} "
            f"NC={(metrics['NC-mean']):.4f} {payload['total-seconds']:.1f}s",
            flush=True,
        )
        torch.cuda.empty_cache()
    if not args.skip_aggregate:
        aggregate(args.output_dir, args.datasets, partial)


if __name__ == "__main__":
    main()
