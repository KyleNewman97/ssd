from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    num_epochs: int

    batch_size: int = Field(default=32)

    # Optimiser parameters
    lr0: float = Field(default=1e-3, description="Initial learning rate")
    momentum: float = Field(default=0.9)
    weight_decay: float = Field(default=5e-4)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)
