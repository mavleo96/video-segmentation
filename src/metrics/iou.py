import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix

from src.utils import one_hot_with_ignore


class SegIoU(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes, ignore_index: int = 255, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Note: We set boundary_ratio to 2% of diagonal as in the Boundary IoU reference
        # Reference: https://arxiv.org/abs/2103.16562
        self.boundary_ratio = 0.02

        self.add_state("conf", torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("b_inter", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("b_union", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def _boundary(self, mask: torch.Tensor, d: int) -> torch.Tensor:
        # Note: pad with 0 so the image border counts as an edge
        padded = F.pad(mask, pad=(d,) * 4, value=0.0)
        eroded = 1 - F.max_pool2d(1 - padded, kernel_size=2 * d + 1, stride=1)
        boundary = mask.bool() & ~eroded.bool()
        return boundary

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        _, H, W = target.shape

        self.conf += confusion_matrix(
            pred, target, task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index
        )

        d = max(1, round(self.boundary_ratio * (H**2 + W**2) ** 0.5))

        pred_oh = one_hot_with_ignore(pred, self.num_classes, self.ignore_index).permute(0, 3, 1, 2)
        target_oh = one_hot_with_ignore(target, self.num_classes, self.ignore_index).permute(0, 3, 1, 2)
        pred_boundary = self._boundary(pred_oh, d)
        target_boundary = self._boundary(target_oh, d)

        valid = (target != self.ignore_index).unsqueeze(1)
        self.b_inter += (pred_boundary & target_boundary & valid).sum((0, 2, 3))
        self.b_union += ((pred_boundary | target_boundary) & valid).sum((0, 2, 3))

    def compute(self) -> dict[str, torch.Tensor]:
        conf = self.conf.double()
        tp = conf.diag()
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)  # 0/0 -> nan for absent classes
        biou = self.b_inter.double() / self.b_union.double()

        return {
            "IoU": iou,
            "mIoU": iou.nanmean(),
            "BIoU": biou,
            "mBIoU": biou.nanmean(),
        }
