import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix


class SegIoU(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: int = 255, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Note: We set boundary_ratio to 2% of diagonal as in the Boundary IoU reference
        # Reference: https://arxiv.org/abs/2103.16562
        self.boundary_ratio = 0.02

        self.add_state(
            "confusion_matrix", torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state("b_inter", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("b_union", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def _boundary(self, x: torch.Tensor, d: int) -> torch.Tensor:
        classes = torch.arange(self.num_classes, device=x.device).view(1, -1, 1, 1)
        mask = (x.unsqueeze(1) == classes).half()

        # Note: pad with 0 so the image border counts as an edge
        padded = F.pad(mask, pad=(d,) * 4, value=0.0)

        # Note: erosion with a square window is separable, so pool rows then columns
        k = 2 * d + 1
        eroded = 1 - F.max_pool2d(F.max_pool2d(1 - padded, kernel_size=(1, k), stride=1), kernel_size=(k, 1), stride=1)
        boundary = mask.bool() & ~eroded.bool()
        return boundary

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        assert pred.ndim == target.ndim == 3
        assert pred.shape == target.shape

        _, H, W = target.shape

        self.confusion_matrix += confusion_matrix(
            pred, target, task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index
        )

        d = max(1, round(self.boundary_ratio * (H**2 + W**2) ** 0.5))

        pred_boundary = self._boundary(pred, d)
        target_boundary = self._boundary(target, d)

        valid = (target != self.ignore_index).unsqueeze(1)
        self.b_inter += (pred_boundary & target_boundary & valid).sum((0, 2, 3))
        self.b_union += ((pred_boundary | target_boundary) & valid).sum((0, 2, 3))

    def compute(self) -> dict[str, torch.Tensor]:
        conf = self.confusion_matrix.float()
        tp = conf.diag()
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)  # 0/0 -> nan for absent classes
        biou = self.b_inter.float() / self.b_union.float()

        return {
            "IoU": iou,
            "mIoU": iou.nanmean(),
            "BIoU": biou,
            "mBIoU": biou.nanmean(),
        }
