"""Precompute RAFT flow between consecutive frames of each val snippet, for the WarpedTC metric.

  python -m scripts.precompute_flow_val

One uncompressed .npz per snippet at $CITYSCAPES_ROOT/flow/val/<city>/<city>_<seq>_<frame>_flow.npz, named after the
labeled frame. Keys, for t = 0..28 (positions in the snippet's frames), each (2, 512, 1024) fp16 at 1/2 res:
  fw_{t} = flow(t, t + 1)
  bw_{t} = flow(t + 1, t)    WarpedTC at frame t + 1: flow_bw = bw_{t}, flow_fw = fw_{t}
Flow convention as in src.flow.warp: flow(src, dst) lives on src's grid and points to where each src pixel is in dst.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import SnippetFrames
from src.flow import RaftFlow

RAFT_SIZE = (512, 1024)  # half of the 1024 x 2048 frame, divisible by 8 as RAFT needs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    out_dir = out_dir / "val"

    # load the snippets from the index
    snippets = json.loads((root / "index" / "cityscapes_sequence_val.json").read_text())["snippets"]
    num_frames = len(snippets[0]["frames"])
    assert all(len(s["frames"]) == num_frames for s in snippets)

    if args.limit:
        snippets = snippets[: args.limit]

    # Note: src_ids / dst_ids index into the snippet's frames, one entry per flow, so both directions of every pair
    # go through RAFT together.
    starts = list(range(num_frames - 1))
    keys = [f"fw_{t}" for t in starts] + [f"bw_{t}" for t in starts]
    src_ids = starts + [t + 1 for t in starts]
    dst_ids = [t + 1 for t in starts] + starts

    raft = RaftFlow(args.device)
    loader = DataLoader(
        SnippetFrames(root, snippets, list(range(num_frames)), RAFT_SIZE),
        batch_size=None,  # one snippet per item, no collation
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_time = time.time()
    pbar = tqdm(loader, desc="val")
    for i, images in pbar:
        snippet = snippets[i]
        flow = raft(images[src_ids], images[dst_ids], args.batch_size)  # (58, 2, 512, 1024), 1/2 res

        # save one npz per snippet, named after the labeled frame
        meta = {
            "model": "raft_large",
            "weights": raft.weights.name,
            "raft_input_hw": list(RAFT_SIZE),
            "storage_scale": 0.5,
            "keys": keys,
            "frames": [f["image"] for f in snippet["frames"]],
        }
        arrays = {key: f.half().cpu().numpy() for key, f in zip(keys, flow, strict=True)}

        label = Path(snippet["label"])  # gtFine/val/<city>/<city>_<seq>_<frame>_gtFine_labelTrainIds.png
        path = out_dir / label.parent.name / label.name.replace("_gtFine_labelTrainIds.png", "_flow.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays, meta=np.array(json.dumps(meta)))

    print(f"{len(snippets)} snippets in {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
