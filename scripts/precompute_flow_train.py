"""Precompute RAFT flow between each train snippet's labeled frame and its neighbors, for the consistency loss.

  python -m scripts.precompute_flow_train                  # labeled frame <-> ±1..±5, all 2975 snippets
  python -m scripts.precompute_flow_train --max-offset 3
  python -m scripts.precompute_flow_train --coverage       # ~100 snippets, offsets ±1..±10, report only, no saving

One uncompressed .npz per snippet at $CITYSCAPES_ROOT/flow/train/<city>/<city>_<seq>_<frame>_flow.npz, named after
the labeled frame L. Keys, for o in -k..-1, 1..k, each (2, 256, 512) fp16 at 1/4 res:
  l2n_{o} = flow(L, L + o)    on L's grid; warp(pred_neighbor, l2n_{o}) brings the neighbor onto L
  n2l_{o} = flow(L + o, L)    for the forward-backward check
Flow convention as in src.flow.warp: flow(src, dst) lives on src's grid and points to where each src pixel is in dst.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import SnippetFrames
from src.flow import RaftFlow, fb_consistency_mask

RAFT_SIZE = (512, 1024)  # half of the 1024 x 2048 frame, divisible by 8 as RAFT needs
COVERAGE_MAX_OFFSET = 10  # the labeled frame is index 19 of 30, so +10 is the furthest future frame
FB_THRESHOLD = 1.0  # forward-backward threshold in pixels of the 1/4-res flow, as in WarpedTC


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-offset", type=int, default=5, help="offsets ±1..±k (k <= 10)")
    parser.add_argument("--coverage", action="store_true", help="report FB coverage per offset ±1..±10, no saving")
    parser.add_argument("--num-snippets", type=int, default=100, help="coverage: snippets sampled from train")
    parser.add_argument("--seed", type=int, default=0, help="coverage: sampling seed")
    parser.add_argument("--batch-size", type=int, default=5, help="flows per RAFT call (5 needs ~5.5 GB)")
    parser.add_argument("--limit", type=int, help="process only the first N snippets (testing)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="output directory (default: $CITYSCAPES_ROOT/flow)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if "CITYSCAPES_ROOT" not in os.environ:
        raise SystemExit("CITYSCAPES_ROOT is not set; conda activate video")
    root = Path(os.environ["CITYSCAPES_ROOT"])
    out_dir = args.out or root / "flow"
    out_dir = out_dir / "train"

    # load the snippets from the index
    snippets = json.loads((root / "index" / "cityscapes_sequence_train.json").read_text())["snippets"]
    label_idx = snippets[0]["label_idx"]
    num_frames = len(snippets[0]["frames"])
    assert all(s["label_idx"] == label_idx and len(s["frames"]) == num_frames for s in snippets)

    if args.coverage:
        rng = np.random.default_rng(args.seed)
        chosen = sorted(rng.choice(len(snippets), args.num_snippets, replace=False))
        snippets = [snippets[i] for i in chosen]
    if args.limit:
        snippets = snippets[: args.limit]

    max_offset = COVERAGE_MAX_OFFSET if args.coverage else args.max_offset
    if not 1 <= max_offset <= min(label_idx, num_frames - 1 - label_idx):
        raise SystemExit(f"--max-offset must be in 1..{num_frames - 1 - label_idx}")

    # Note: frame_ids are the snippet frames each item decodes, labeled frame first; src_ids / dst_ids index into
    # them, one entry per flow, so both directions of every pair go through RAFT together.
    offsets = [o for o in range(-max_offset, max_offset + 1) if o != 0]
    num_offsets = len(offsets)
    neighbor_ids = list(range(1, num_offsets + 1))

    frame_ids = [label_idx] + [label_idx + o for o in offsets]
    keys = [f"l2n_{o}" for o in offsets] + [f"n2l_{o}" for o in offsets]
    src_ids = [0] * num_offsets + neighbor_ids
    dst_ids = neighbor_ids + [0] * num_offsets

    raft = RaftFlow(args.device)
    loader = DataLoader(
        SnippetFrames(root, snippets, frame_ids, RAFT_SIZE),
        batch_size=None,  # one snippet per item, no collation
        num_workers=args.num_workers,
        pin_memory=True,
    )

    coverage, magnitude = [], []  # coverage: per snippet, one value per offset
    start_time = time.time()
    pbar = tqdm(loader, desc="coverage" if args.coverage else "train")
    for i, images in pbar:
        snippet = snippets[i]

        flow = raft(images[src_ids], images[dst_ids], args.batch_size)  # 1/2 res

        # Note: average 2x2 blocks for 1/4 res, and halve the vectors so they stay in pixels of the smaller grid.
        flow = F.avg_pool2d(flow, kernel_size=2) / 2  # (2 * num_offsets, 2, 256, 512)

        if args.coverage:
            l2n, n2l = flow[:num_offsets], flow[num_offsets:]
            valid = fb_consistency_mask(l2n, n2l, FB_THRESHOLD)
            coverage.append(valid.float().mean((1, 2)).cpu().numpy())
            magnitude.append(l2n.norm(dim=1).flatten(1).median(dim=1).values.cpu().numpy())
            continue

        # save one npz per snippet, named after the labeled frame
        meta = {
            "model": "raft_large",
            "weights": raft.weights.name,
            "raft_input_hw": list(RAFT_SIZE),
            "storage_scale": 0.25,
            "keys": keys,
            "labeled": snippet["image"],
            "neighbors": {str(o): snippet["frames"][label_idx + o]["image"] for o in offsets},
        }
        arrays = {key: f.half().cpu().numpy() for key, f in zip(keys, flow, strict=True)}

        label = Path(snippet["label"])  # gtFine/train/<city>/<city>_<seq>_<frame>_gtFine_labelTrainIds.png
        path = out_dir / label.parent.name / label.name.replace("_gtFine_labelTrainIds.png", "_flow.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays, meta=np.array(json.dumps(meta)))

    print(f"{len(snippets)} snippets in {(time.time() - start_time) / 60:.1f} min")

    if not args.coverage:
        return

    # Note: split snippets by timing; a jump (any interval > 1.5x the median) makes offsets across it span far more
    # time. Frame 0's timestamp is 0 (missing) in short sequences, so it is left out.
    irregular = []
    for snippet in snippets:
        timestamps = np.array([int((root / f["timestamp"]).read_text()) for f in snippet["frames"][1:]])
        intervals = np.diff(timestamps)
        irregular.append(intervals.max() > 1.5 * np.median(intervals))
    irregular = np.array(irregular)

    coverage, magnitude = np.stack(coverage), np.stack(magnitude)
    groups = {"all": np.ones_like(irregular), "uniform": ~irregular, "jumps": irregular}

    # print the coverage table: mean and 10th percentile per offset and timing group
    print(f"\nFB-consistent and in-bounds fraction at 1/4 res (threshold {FB_THRESHOLD} px)")
    print(f"{len(snippets)} snippets: {(~irregular).sum()} uniform timing, {irregular.sum()} with jumps/skips")
    header = " | ".join(f"{name:>8} {'p10':>6}" for name in groups)
    print(f"{'offset':>6} | {header} | median |flow| px")

    for col, offset in enumerate(offsets):
        cells = []
        for mask in groups.values():
            values = coverage[mask, col]
            cells.append(f"{values.mean():8.3f} {np.percentile(values, 10):6.3f}" if len(values) else f"{'-':>15}")
        print(f"{offset:>6} | {' | '.join(cells)} | {np.median(magnitude[:, col]):6.1f}")


if __name__ == "__main__":
    main()
