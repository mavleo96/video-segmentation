import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.backbone_utils import load_backbone
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.beit.modeling_beit import BeitFCNHead, BeitFPNNeck, BeitUperHead

from .configuration import DINOv3ForSemanticSegmentationConfig


# Note: We replace the batchnorm to group norm for 2 reasons:
# 1. Training with consitency loss will have similar images in same batch, not ideal for BN
# 2. Batch size is small, BN statistics are noisy
def bn_to_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """Recursively replace BatchNorm2d with GroupNorm"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            bn_to_gn(child, num_groups)
    return module


class DINOv3ForSemanticSegmentation(PreTrainedModel):
    config_class = DINOv3ForSemanticSegmentationConfig
    main_input_name = "pixel_values"

    def __init__(self, config: DINOv3ForSemanticSegmentationConfig) -> None:
        super().__init__(config)
        if len(config.dinov3_config.out_indices) != 4:
            raise ValueError(
                "DINOv3ForSemanticSegmentationConfig requires config.dinov3_config.out_indices to be a list of 4 "
                "integers, specifying which features to use from the backbone."
            )

        self.backbone = load_backbone(config.dinov3_config)

        self.fpn_neck = bn_to_gn(BeitFPNNeck(config), config.num_groups)

        # Semantic segmentation head(s)
        self.decode_head = bn_to_gn(BeitUperHead(config), config.num_groups)
        self.auxiliary_head = bn_to_gn(BeitFCNHead(config), config.num_groups) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SemanticSegmenterOutput:
        _, _, H, W = pixel_values.shape

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        feature_maps = self.backbone(pixel_values).feature_maps  # [(b, d, h, w)] * 4
        feature_maps = self.fpn_neck(feature_maps)  # h, w -> (4x, 2x, 1x, 0.5x)

        logits = self.decode_head(feature_maps)  # (b, c, 4h, 4w)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)  # h, w -> H, W

        auxiliary_logits = None
        if self.auxiliary_head is not None and self.training:
            auxiliary_logits = self.auxiliary_head(feature_maps)
            auxiliary_logits = F.interpolate(auxiliary_logits, size=(H, W), mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # TODO: check if we need to apply loss at lower resolution to save memory
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            loss = loss_fct(logits, labels)
            if auxiliary_logits is not None and self.config.use_auxiliary_head:
                loss = loss + self.config.auxiliary_loss_weight * loss_fct(auxiliary_logits, labels)

        # TODO: check if we need to return hidden_states and attentions
        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
        )


__all__ = ["DINOv3ForSemanticSegmentation"]
