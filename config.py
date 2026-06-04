from dataclasses import dataclass, field
from typing import List


@dataclass
class ProbeConfig:
    # Data
    dataset: str = "imagenet_c"  # "imagenet_c" | "cifar10_c"
    data_root: str = "/Dataset/yezhong"
    corruption_families: List[str] = field(default_factory=lambda: [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ])
    severities: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    batch_size: int = 64
    num_workers: int = 2
    max_batches: int = 0  # 0 means no limit; useful for debug smoke tests

    # Model
    model_name: str = "resnet50"  # "resnet50" | "resnet101" | "vit_b16"
    source_stats_path: str = "/data/source_stats"
    device: str = "cuda"

    # Diagnostic optimization
    opt_steps: int = 100
    opt_lr: float = 0.01
    lambda1: float = 1.0
    lambda2: float = 0.0

    # SPA
    freq_bins: int = 64
    low_freq_cutoff_bin: int = 16

    # ActMAD
    actmad_layers: List[str] = field(default_factory=list)
    min_batch_size_for_actmad: int = 8

    # Output
    output_dir: str = "./outputs"
    save_xstar: bool = False
    save_delta: bool = True
