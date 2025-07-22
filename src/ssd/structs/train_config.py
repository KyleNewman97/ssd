from pathlib import Path
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    num_epochs: int
    anchor_iou_threshold: float = Field(
        default=0.1,
        description=(
            "The IoU threshold at which an anchor is considered to match a GT box."
        ),
    )

    batch_size: int = Field(default=32)

    # Optimiser parameters
    lr0: float = Field(default=1e-3, description="Initial learning rate")
    momentum: float = Field(default=0.9)
    weight_decay: float = Field(default=5e-4)

    # Loss parameters
    box_loss_scaling_factor: float = Field(default=6)

    # Dataset parameters
    num_classes: int
    train_images_dir: Path
    train_labels_dir: Path
    val_images_dir: Path
    val_labels_dir: Path

    # Data transform parameters
    image_width: int = Field(default=300)
    image_height: int = Field(default=300)
    dtype: torch.dtype = Field(default=torch.float32)

    # Logging configuration
    log_dir: Path
    experiment_name: str = Field(default_factory=lambda: f"{uuid4()}")
    project_name: str = Field(default="ssd")
    team_name: str

    # Metrics calculation
    min_confidence_threshold: float = Field(default=0.1)
    num_top_k: int = Field(default=200)
    nms_iou_threshold: float = Field(default=0.4)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def experiment_dir(self) -> Path:
        return self.log_dir / self.experiment_name
