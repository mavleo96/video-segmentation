import torch
import torch.nn.functional as F


def one_hot_with_ignore(x: torch.Tensor, num_classes: int, ignore_label: int) -> torch.Tensor:
    """A thin wrapper around F.one_hot that ignores the ignore_label."""
    mask = x == ignore_label

    safe_x = x.clone()
    safe_x[mask] = 0

    oh = F.one_hot(safe_x, num_classes)
    oh[mask] = 0

    return oh
