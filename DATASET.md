# Datasets
> Generated with [Claude Code](https://claude.com/claude-code). Every number below was measured on the downloaded data.

Both datasets live under `$DATA_ROOT` (`/data/video-segmentation`); the `video` conda env sets the paths.

| | Cityscapes-sequence | KITTI-STEP |
|---|---|---|
| Root | `$CITYSCAPES_ROOT` | `$KITTISTEP_ROOT` |
| Download | `bash scripts/download_cityscapes.sh` | `bash scripts/download_kittistep.sh` |
| Disk | 229 GB | 6.2 GB |
| Image size (w × h) | 2048 × 1024 (all frames) | 4 sizes, ~1242 × 375 ([details](#image-sizes)) |
| Frame rate | 17 Hz (with exceptions, see [Frame timing](#frame-timing)) | 10 Hz |
| Train | 2975 clips × 30 = 89250 frames | 12 sequences, 5027 frames |
| Val | 500 clips × 30 = 15000 frames | 9 sequences, 2981 frames |
| Labeled frames | 1 per clip (2975 / 500) | every frame (5027 / 2981) |
| Labels | semantic (`labelTrainIds`) | panoptic (semantic + track ID) |

Both use the same 19 Cityscapes train-ID classes with **255 = ignore**. Test splits are not downloaded (no public
labels). Both download scripts are resumable; rerun after an interruption.

**Contents:** [Classes](#classes) · [Cityscapes-sequence](#cityscapes-sequence) · [KITTI-STEP](#kitti-step) ·
[Index files](#index-files) · [Training notes](#training-notes)

## Classes

Names, IDs and colors come from `cityscapesscripts.helpers.labels`. Pixel % is each class's share of all pixels in
the train split (255 included).

| Train ID | Class | Cityscapes `labelId` | Color (RGB) | Thing | Cityscapes px % | KITTI-STEP px % |
|---|---|---|---|---|---|---|
| 0 | road | 7 | 128, 64, 128 | | 32.63 | 18.96 |
| 1 | sidewalk | 8 | 244, 35, 232 | | 5.39 | 7.04 |
| 2 | building | 11 | 70, 70, 70 | | 20.19 | 15.88 |
| 3 | wall | 12 | 102, 102, 156 | | 0.58 | 0.45 |
| 4 | fence | 13 | 190, 153, 153 | | 0.78 | 1.53 |
| 5 | pole | 17 | 153, 153, 153 | | 1.09 | 1.05 |
| 6 | traffic light | 19 | 250, 170, 30 | | 0.18 | 0.19 |
| 7 | traffic sign | 20 | 220, 220, 0 | | 0.49 | 0.58 |
| 8 | vegetation | 21 | 107, 142, 35 | | 14.08 | 27.71 |
| 9 | terrain | 22 | 152, 251, 152 | | 1.03 | 4.80 |
| 10 | sky | 23 | 70, 130, 180 | | 3.55 | 8.77 |
| 11 | person | 24 | 220, 20, 60 | ✓ | 1.08 | 1.20 |
| 12 | rider | 25 | 255, 0, 0 | ✓ | 0.12 | 0.10 |
| 13 | car | 26 | 0, 0, 142 | ✓ | 6.20 | 5.92 |
| 14 | truck | 27 | 0, 0, 70 | ✓ | 0.24 | 0.89 |
| 15 | bus | 28 | 0, 60, 100 | ✓ | 0.21 | 0.49 |
| 16 | train | 31 | 0, 80, 100 | ✓ | 0.21 | 0.17 |
| 17 | motorcycle | 32 | 0, 0, 230 | ✓ | 0.09 | 0.03 |
| 18 | bicycle | 33 | 119, 11, 32 | ✓ | 0.37 | 0.61 |
| 255 | ignore | 0–6, 9, 10, 14–16, 18, 29, 30 | — | | 11.52 | 3.64 |

- **Ignore (255)** covers the void classes (unlabeled, ego vehicle, rectification border, out of roi, static,
  dynamic, ground) and the classes excluded from evaluation (parking, rail track, guard rail, bridge, tunnel,
  polegroup, caravan, trailer). Exclude it from the loss (`ignore_index=255`) and from mIoU.
- **Thing** = the class has instance annotations in Cityscapes (train IDs 11–18). In KITTI-STEP only person and car
  do ([Panoptic labels](#panoptic-labels)).
- **Imbalance:** road + building + vegetation are ~67% of Cityscapes pixels; rider, motorcycle, traffic light,
  truck, bus and train are each < 0.3%. Cityscapes val shares are within ~1 point of train.
- KITTI-STEP has more vegetation, sky and terrain and less road and ignore than Cityscapes.

## Cityscapes-sequence

### Layout

```
cityscapes/
  leftImg8bit_sequence/{train,val}/<city>/<city>_<seq>_<frame>_leftImg8bit.png   # RGB frames
  gtFine/{train,val,test}/<city>/<city>_<seq>_<frame>_gtFine_*                    # labels, see below
  timestamp_sequence/{train,val,test}/<city>/<city>_<seq>_<frame>_timestamp.txt   # one integer, ns
```

- Train: 18 cities. Val: `frankfurt`, `lindau`, `munster`.
- `gtFine/test` and `timestamp_sequence/test` came with their zips; test frames were not downloaded.

### Clips

- A clip is 30 consecutive frames around one labeled frame, which is the **20th frame (index 19)**: frames
  −19 … +10.
- Frame numbers are absolute within a recording, not 0–29: `frankfurt_000000_000294` is labeled and its clip is
  frames `000275`–`000304`. To find a label's frame, swap `gtFine/` → `leftImg8bit_sequence/` and
  `_gtFine_labelTrainIds` → `_leftImg8bit`.
- Most recordings (`<city>_<seq>`) are exactly one 30-frame clip. 15 are long continuous recordings (210–7440
  frames) with several labeled frames: 8 in train, 2 in val, 5 in test. So build clips from labeled frames,
  **not** by grouping on `<city>_<seq>`.
- In train and val every labeled frame has its full 30-frame window on disk, every frame on disk is in exactly one
  clip, and clips never overlap (labeled frames in one recording are ≥ 31 frames apart in train, ≥ 39 in val).

Val labeled frames by city:

| City | Labeled frames | Recordings | Labeled frame numbers |
|---|---|---|---|
| `munster` | 174 | 174 short | all `000019` |
| `lindau` | 59 | 59 short | all `000019` |
| `frankfurt` | 267 | 2 long | `frankfurt_000000`: 50, `000294`–`022797`; `frankfurt_000001`: 217, `000538`–`083852` |

### Frame timing

- The nominal frame interval is 58.8 ms (17 Hz).
- Timestamp `0` means missing. It appears on the first frame of every short recording (1877 train / 233 val clips)
  and nowhere else.
- Not every clip is evenly spaced. Comparing consecutive timestamps (frames 1–29; frame 0 is unknown in short
  clips):

  | Clip timing | Train | Val |
  |---|---|---|
  | uniform 17 Hz | 2536 | 434 |
  | 1–5 jumps (mostly one, between frames 20 and 21) | 358 | 65 (`frankfurt`) |
  | skipped frames throughout (intervals of 2–3 × 58.8 ms) | 81 (`aachen`) | 1 (`lindau`) |

- Jumps range from 2× to 31× the nominal interval. A clip spans 1.71 s at the median, up to 5.25 s in train
  (2.8 s in val).

### Label files

Each labeled frame has five files in `gtFine/`:

| Suffix | Format | Content |
|---|---|---|
| `_gtFine_labelTrainIds.png` | uint8 | train ID 0–18 / 255; what we train on (made by `csCreateTrainIdLabelImgs`) |
| `_gtFine_labelIds.png` | uint8 | original `labelId` 0–33 |
| `_gtFine_instanceIds.png` | uint16 | `labelId × 1000 + k` for each object; plain `labelId` for stuff and crowd regions |
| `_gtFine_color.png` | RGBA | visualization using the colors above |
| `_gtFine_polygons.json` | JSON | source polygons the PNGs were rasterized from |

- Instances exist for `labelId` 24–33: the 8 thing classes plus the ignored caravan and trailer.
- Crowd regions (a class but no instance) occur for person, rider, car and bicycle, and for motorcycle in train.
- Instance IDs are per image; they are not tracked across frames.

## KITTI-STEP

### Layout

```
kittistep/
  images/{train,val}/<seq>/<frame>.png          # RGB, KITTI tracking image_02 (left camera)
  panoptic_maps/{train,val}/<seq>/<frame>.png   # RGB-encoded panoptic label, same name as its image
```

- Train sequences: `0000 0001 0003 0004 0005 0009 0011 0012 0015 0017 0019 0020`.
- Val sequences: `0002 0006 0007 0008 0010 0013 0014 0016 0018`.
- Frames are numbered contiguously from `000000` within each sequence, and every image has a label.

### Image sizes

The size is fixed within a sequence but differs between sequences; a label always has the same size as its image.

| w × h | Train frames | Val frames |
|---|---|---|
| 1242 × 375 | 2610 | 2327 |
| 1238 × 374 | 1059 | 339 |
| 1241 × 376 | 837 | — |
| 1224 × 370 | 521 | 315 |

### Panoptic labels

- 8-bit RGB PNG: `semantic = R` (train ID), `instance = G * 256 + B` (track ID, 0 = none). A common combined ID is
  `semantic * 1000 + instance`. Cast to `int` before decoding; `G * 256` overflows uint8.
- Only **person (11)** and **car (13)** carry track IDs. All other thing classes (rider, truck, bus, …) are
  semantic-only.
- About 15% of person/car pixels have instance 0: regions with a class but no track (crowd or unresolved).
- Track IDs are stable over time within a sequence, and each ID belongs to a single class. IDs restart in each
  sequence (from 1 in most), so the same ID in two sequences is a different object.

| | Train | Val |
|---|---|---|
| Tracks (person / car) | 530 (99 / 431) | 219 (68 / 151) |
| Max track ID in a sequence | 134 | 68 |
| Frames with ≥ 1 track | 4520 / 5027 | 2642 / 2981 |

## Index files

`python -m src.data.index` builds one JSON index per dataset and split for the dataset classes, written to
`scripts/train/<dataset>_<split>.json`. Only Cityscapes is implemented so far (`cityscapes_seq_{train,val}.json`).

```json
{
  "dataset": "cityscapes_seq",
  "split": "train",
  "label_format": "semantic_trainid",
  "fps": 17,
  "num_clips": 2975,
  "num_frames": 89250,
  "num_labeled": 2975,
  "clips": [
    {
      "clip": "aachen_000000_000019",
      "frames": ["leftImg8bit_sequence/train/aachen/aachen_000000_000000_leftImg8bit.png", "..."],
      "labels": [null, "...", "gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png", "...", null]
    }
  ]
}
```

- Paths are relative to the dataset root (`$CITYSCAPES_ROOT`).
- The clip ID is the labeled frame's name, which stays unique in the long recordings.
- `labels` runs parallel to `frames`, with `null` for unlabeled frames.
- `label_format` tells the loader how to decode labels (`semantic_trainid`, later `kitti_step_panoptic`).
- While building, the script checks that every frame exists, that clips don't overlap, that each label matches its
  frame's size, and that no frame on disk is left out.

## Training notes

- **Aspect ratio:** Cityscapes is 2:1 and KITTI-STEP ≈ 3.3:1. KITTI-STEP needs a common size to batch (resize or
  crop; e.g. 1216 × 368 fits inside the smallest size, 1224 × 370).
- **Frame rate:** 17 Hz vs 10 Hz, so the same frame offset covers different time and motion. Within Cityscapes,
  ~15% of clips also have jumps or skipped frames ([Frame timing](#frame-timing)), so a fixed offset is not a fixed
  time step there either.
- **Cityscapes val clips are not independent:** 267 of 500 come from two continuous Frankfurt drives, with
  neighboring clips as little as ~2 s apart, so scenes repeat. mIoU is unaffected, but keep it in mind when
  sub-splitting val or reading per-clip results.
