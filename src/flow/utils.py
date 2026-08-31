import torch
import torch.nn.functional as F


def warp(x: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x at p + flow(p) for each pixel p. x: (B, C, H, W), flow: (B, 2, H, W) in pixels.

    Returns the warped x and the sampling positions (B, 2, H, W). Differentiable in x and flow (grid_sample), so it
    also serves a warped consistency loss; out-of-bounds samples are 0, so mask them with the positions.
    """
    assert x.ndim == 4 and flow.ndim == 4
    assert x.shape[0] == flow.shape[0] and x.shape[2:] == flow.shape[2:]
    assert flow.shape[1] == 2

    _, _, H, W = flow.shape

    # create a meshgrid and add flow to it
    ys, xs = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing="ij")
    pos = torch.stack((xs, ys)).float() + flow

    # normalize the grid and sample x at calculated positions
    grid = torch.stack((2 * pos[:, 0] / (W - 1) - 1, 2 * pos[:, 1] / (H - 1) - 1), dim=-1)
    out = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=True)

    return out, pos


def fb_consistency_mask(flow_ab: torch.Tensor, flow_ba: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """(B, H, W) True where flow_ab lands in-bounds and passes the forward-backward check.

    flow_ab: a -> b on a's grid, flow_ba: b -> a on b's grid, both (B, 2, H, W) in pixels. Following flow_ab and then
    flow_ba should return within `threshold` px of the start; pixels that don't are occluded or have unreliable flow.
    """
    assert flow_ab.shape == flow_ba.shape and flow_ab.ndim == 4

    _, _, H, W = flow_ab.shape

    flow_ba_at, pos = warp(flow_ba, flow_ab)
    consistent = (flow_ab + flow_ba_at).norm(dim=1) < threshold
    in_bounds = (pos[:, 0] >= 0) & (pos[:, 0] <= W - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= H - 1)

    return consistent & in_bounds
