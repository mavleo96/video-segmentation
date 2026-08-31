"""Build the JSON indexes that the dataset classes read.

  python -m src.data.index                         # cityscapes train + val -> $CITYSCAPES_ROOT/index/
  python -m src.data.index --dataset kitti_step    # kitti_step val -> $KITTISTEP_ROOT/index/
  python -m src.data.index --splits val --out DIR

Indexes are written to <dataset root>/index/<dataset>_<split>.json. Paths in an index are relative to the dataset
root ($CITYSCAPES_ROOT / $KITTISTEP_ROOT), so moving the data only needs the env var updated, not a rebuild.
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

CS_SNIPPET_LEN = 30  # frames per snippet
CS_LABELED = 19  # position of the labeled frame within its snippet


def build_cityscapes(root: Path, split: str) -> dict:
    labels = sorted((root / "gtFine" / split).glob("*/*_gtFine_labelTrainIds.png"))
    if not labels:
        raise FileNotFoundError(f"no labelTrainIds under {root}/gtFine/{split}; run csCreateTrainIdLabelImgs")

    # Note: list the frames once rather than stat-ing each path; the set also lets us check below that every
    # frame on disk ends up in exactly one snippet.
    seq_dir = root / "leftImg8bit_sequence" / split
    on_disk = {p.relative_to(root).as_posix() for p in seq_dir.glob("*/*_leftImg8bit.png")}
    ts_dir = root / "timestamp_sequence" / split
    ts_on_disk = {p.relative_to(root).as_posix() for p in ts_dir.glob("*/*_timestamp.txt")}

    snippets, used = [], set()
    pbar = tqdm(labels, desc=f"cityscapes {split}")
    for label in pbar:
        city, seq, num = label.name.split("_")[:3]

        # Note: frame numbers are absolute within a sequence, and long sequences hold several labeled frames,
        # so each snippet is built around its labeled frame, not by grouping on <city>_<seq>.
        start = int(num) - CS_LABELED
        images = [
            f"leftImg8bit_sequence/{split}/{city}/{city}_{seq}_{start + i:06d}_leftImg8bit.png"
            for i in range(CS_SNIPPET_LEN)
        ]
        timestamps = [
            f"timestamp_sequence/{split}/{city}/{city}_{seq}_{start + i:06d}_timestamp.txt"
            for i in range(CS_SNIPPET_LEN)
        ]

        missing = [f for f in images if f not in on_disk] + [t for t in timestamps if t not in ts_on_disk]
        if missing:
            raise FileNotFoundError(f"{label.name}: {len(missing)} frames/timestamps missing, e.g. {missing[0]}")
        if used.intersection(images):
            raise ValueError(f"{label.name}: snippet overlaps another snippet")
        used.update(images)

        # Note: Image.open reads only the header, so getting the size doesn't decode the PNG.
        size, label_size = Image.open(root / images[CS_LABELED]).size, Image.open(label).size
        if label_size != size:
            raise ValueError(f"{label.name}: label size {label_size} != frame size {size}")

        snippets.append(
            {
                "image": images[CS_LABELED],
                "label": label.relative_to(root).as_posix(),
                "label_idx": CS_LABELED,
                "frames": [{"image": im, "timestamp": ts} for im, ts in zip(images, timestamps, strict=True)],
            }
        )

    unused = on_disk - used
    if unused:
        raise ValueError(f"{len(unused)} frames on disk belong to no snippet, e.g. {min(unused)}")

    return {
        "dataset": "cityscapes_sequence",
        "split": split,
        "label_format": "semantic_trainid",
        "num_snippets": len(snippets),
        "num_frames": sum(len(s["frames"]) for s in snippets),
        "num_labeled": len(snippets),  # one labeled frame per snippet
        "snippets": snippets,
    }


def build_kitti_step(root: Path, split: str) -> dict:
    # Note: the split's sequence list comes from the label folders (the official STEP train/val split); the
    # image folders must hold exactly the same sequences.
    seq_ids = sorted(p.name for p in (root / "panoptic_maps" / split).iterdir() if p.is_dir())
    if not seq_ids:
        raise FileNotFoundError(f"no sequences under {root}/panoptic_maps/{split}")
    image_ids = sorted(p.name for p in (root / "images" / split).iterdir() if p.is_dir())
    if image_ids != seq_ids:
        raise ValueError(f"{split}: image sequences {image_ids} != label sequences {seq_ids}")

    sequences = []
    pbar = tqdm(seq_ids, desc=f"kitti_step {split}")
    for seq_id in pbar:
        names = sorted(p.name for p in (root / "images" / split / seq_id).glob("*.png"))
        label_names = sorted(p.name for p in (root / "panoptic_maps" / split / seq_id).glob("*.png"))

        if names != label_names:
            raise ValueError(f"{seq_id}: image and label file names differ")
        # Note: there are no timestamps, so frame order and spacing come only from the frame numbers;
        # require them to run 000000, 000001, ... with no gaps.
        if names != [f"{i:06d}.png" for i in range(len(names))]:
            raise ValueError(f"{seq_id}: frame numbers are not contiguous from 000000")

        frames = []
        for name in names:
            image = f"images/{split}/{seq_id}/{name}"
            label = f"panoptic_maps/{split}/{seq_id}/{name}"
            if Image.open(root / image).size != Image.open(root / label).size:  # header only
                raise ValueError(f"{label}: label size differs from its image")
            frames.append({"image": image, "label": label})

        sequences.append({"frames": frames})

    num_frames = sum(len(s["frames"]) for s in sequences)
    return {
        "dataset": "kitti_step",
        "split": split,
        "label_format": "kitti_step_panoptic",
        "num_sequences": len(sequences),
        "num_frames": num_frames,
        "num_labeled": num_frames,  # every frame is labeled
        "sequences": sequences,
    }


# Note: KITTI-STEP is only used for validation, so its train split is not indexed by default.
BUILDERS = {  # name: (builder, root env var, default splits)
    "cityscapes": (build_cityscapes, "CITYSCAPES_ROOT", ["train", "val"]),
    "kitti_step": (build_kitti_step, "KITTISTEP_ROOT", ["val"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=BUILDERS, default="cityscapes")
    parser.add_argument("--splits", nargs="+", help="default: cityscapes train val, kitti_step val")
    parser.add_argument("--out", type=Path, help="output directory (default: <dataset root>/index)")
    args = parser.parse_args()

    build, root_env, default_splits = BUILDERS[args.dataset]
    if root_env not in os.environ:
        raise SystemExit(f"{root_env} is not set; conda activate video")
    root = Path(os.environ[root_env])

    out_dir = args.out or root / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits or default_splits:
        index = build(root, split)

        # Note: write to a temp file and rename, so an interrupted run never leaves a truncated index behind.
        path = out_dir / f"{index['dataset']}_{index['split']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(path)

        unit = "snippets" if "snippets" in index else "sequences"
        print(f"{path}: {index[f'num_{unit}']} {unit}, {index['num_frames']} frames, {index['num_labeled']} labeled")


if __name__ == "__main__":
    main()
