from transformers import DINOv3ViTConfig
from transformers.configuration_utils import PreTrainedConfig, SubConfigSpec


class DINOv3ForSemanticSegmentationConfig(PreTrainedConfig):
    model_type = "beit_dinov3"

    # backbone config
    sub_configs_defaults = {
        "dinov3_config": SubConfigSpec(
            config_class=DINOv3ViTConfig,
            model_type="dinov3_vit",
            init_kwargs={
                "out_indices": [4, 6, 8, 12],
                "apply_layernorm": True,
            },
        ),
    }
    dinov3_config: DINOv3ViTConfig | None = None

    initializer_range: float = 0.02
    num_groups: int = 32

    # upernet head
    pool_scales: tuple[int, ...] = (1, 2, 3, 6)
    num_labels: int = 19

    # auxiliary head
    use_auxiliary_head: bool = True
    auxiliary_loss_weight: float = 0.4
    auxiliary_in_channels: int | None = None
    auxiliary_channels: int = 256
    auxiliary_num_convs: int = 1
    auxiliary_concat_input: bool = False

    semantic_loss_ignore_index: int = 255

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # hidden size for neck and upernet head copied from backbone
        self.hidden_size = self.dinov3_config.hidden_size


__all__ = ["DINOv3ForSemanticSegmentationConfig"]
