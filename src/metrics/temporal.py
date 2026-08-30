from collections import deque

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix

from src.utils import warp


class WarpedTC(Metric):
    """
    Warped Temporal Consistency: mean IoU between the prediction at t and the prediction at t-1 warped by optical flow.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes: int, fb_thresh: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fb_thresh = fb_thresh

        self.add_state(
            "confusion_matrix", torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state("valid_pixels", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total_pixels", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred_t: Tensor, pred_prev: Tensor, flow_bw: Tensor, flow_fw: Tensor) -> None:
        """pred_t, pred_prev: (B, H, W) class ids. flow_bw: t -> t-1, flow_fw: t-1 -> t, (B, 2, H, W) in pixels."""
        _, H, W = pred_t.shape

        # warp the previous prediction using the backward flow
        warped, pos = warp(pred_prev.unsqueeze(1).float(), flow_bw, "nearest")
        warped = warped.squeeze(1).long()

        # Note: following flow_bw and then flow_fw should return within fb_thresh px of the start; pixels that
        # don't are occluded or have unreliable flow. pos is shared with the warp above (same flow_bw).
        fw_at, _ = warp(flow_fw, flow_bw, "bilinear")
        consistent = (flow_bw + fw_at).norm(dim=1) < self.fb_thresh
        in_bounds = (pos[:, 0] >= 0) & (pos[:, 0] <= W - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= H - 1)
        valid = consistent & in_bounds

        self.confusion_matrix += confusion_matrix(
            warped[valid],
            pred_t[valid],
            task="multiclass",
            num_classes=self.num_classes,
            validate_args=False,
        )
        self.valid_pixels += valid.sum()
        self.total_pixels += valid.numel()

    def compute(self) -> dict[str, Tensor]:
        conf = self.confusion_matrix.float()
        tp = conf.diag()
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)  # 0/0 -> nan for absent classes

        return {
            "TC": iou.nanmean(),
            "TC_per_class": iou,
            "coverage": self.valid_pixels.float() / self.total_pixels,
        }


class VEC(Metric):
    """Video Effective Consistency (DTERN, ICCV 2025), sliding window, accumulated over the dataset.

    Over each window of `window` consecutive frames, a pixel is stable in the ground truth (or prediction) if
    its class never changes; per class, VEC is the IoU between the stable-and-correct pixels. Needs labels on
    every frame, so it applies to KITTI-STEP, not Cityscapes (one labeled frame per snippet).

    Call update(pred, target) once per frame, in order, and new_sequence() before each new sequence. Use
    update(), not metric(...): the window lives outside the metric state, and Metric.forward resets state
    around each call. Under DDP, each sequence must be evaluated whole on one process.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes: int, window: int = 8, ignore_index: int = 255, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.window = window
        self.ignore_index = ignore_index

        self.add_state("inter", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("gt_count", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("pr_count", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self._buf = deque(maxlen=window)

    def forward(self, *args, **kwargs):
        raise RuntimeError("VEC keeps a sliding window across calls; use update(), not metric(...)")

    def new_sequence(self) -> None:
        self._buf.clear()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        assert pred.ndim == target.ndim == 2
        assert pred.shape == target.shape

        self._buf.append((pred, target))
        if len(self._buf) < self.window:
            return

        preds = torch.stack([p for p, _ in self._buf])  # (n, H, W)
        gts = torch.stack([g for _, g in self._buf])
        g0, p0 = gts[0], preds[0]

        valid = (gts != self.ignore_index).all(0)
        gt_stable = (gts == g0).all(0) & valid
        pr_stable = (preds == p0).all(0) & valid
        both = gt_stable & pr_stable & (g0 == p0)

        C = self.num_classes
        self.inter += torch.bincount(g0[both], minlength=C)[:C]
        self.gt_count += torch.bincount(g0[gt_stable], minlength=C)[:C]
        self.pr_count += torch.bincount(p0[pr_stable], minlength=C)[:C]

    def compute(self) -> dict[str, torch.Tensor]:
        inter = self.inter.float()
        union = self.gt_count.float() + self.pr_count.float() - inter
        iou = inter / union  # 0/0 -> nan for absent classes

        return {
            "VEC": iou.nanmean(),
            "VEC_per_class": iou,
        }

    def reset(self) -> None:
        super().reset()
        self._buf.clear()
