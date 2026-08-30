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

        self.add_state("conf", torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("valid_px", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total_px", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

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

        self.conf += confusion_matrix(
            warped[valid],
            pred_t[valid],
            task="multiclass",
            num_classes=self.num_classes,
            validate_args=False,
        )
        self.valid_px += valid.sum()
        self.total_px += valid.numel()

    def compute(self) -> dict[str, Tensor]:
        conf = self.conf.float()
        tp = conf.diag()
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)  # 0/0 -> nan for absent classes

        return {
            "TC": iou.nanmean(),
            "TC_per_class": iou,
            "coverage": self.valid_px.float() / self.total_px,
        }
