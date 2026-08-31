import torch
import torch._inductor.config
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class RaftFlow:
    def __init__(self, device: str = "cuda"):
        # Note: inductor's compile worker pool fails here ("Attempting to use a closed pool"), so compile in the
        # main process
        torch._inductor.config.compile_threads = 1

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = torch.compile(raft_large(weights=self.weights).eval().to(device), dynamic=False)
        self.device = device

    @torch.inference_mode()
    def __call__(self, src: torch.Tensor, dst: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Flow src -> dst. src, dst: (B, 3, H, W) uint8, H and W divisible by 8 -> (B, 2, H, W) float32 in pixels."""
        assert src.shape == dst.shape and src.ndim == 4
        assert src.shape[2] % 8 == 0 and src.shape[3] % 8 == 0

        # Note: pad the last chunk to batch_size by repeating the last pair, so every call has the same shape; a new
        # batch size makes torch.compile recompile. The compiled model keeps ~0.36 GB per flow in the batch allocated
        # between calls on top of its working memory, so on 12 GB keep batch_size <= 6.
        num_flows = len(src)
        pad = -num_flows % batch_size
        if pad:
            src = torch.cat([src, src[-1:].expand(pad, -1, -1, -1)])
            dst = torch.cat([dst, dst[-1:].expand(pad, -1, -1, -1)])

        # run RAFT in chunks of batch_size flows; the last refinement is the final flow
        flows = []
        for i in range(0, len(src), batch_size):
            src_batch = src[i : i + batch_size].to(self.device).float() / 255
            dst_batch = dst[i : i + batch_size].to(self.device).float() / 255
            src_batch, dst_batch = self.transforms(src_batch, dst_batch)
            flows.append(self.model(src_batch, dst_batch)[-1])

        return torch.cat(flows)[:num_flows]
