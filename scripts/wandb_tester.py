from pathlib import Path
from uuid import uuid4

import numpy as np

from ssd.structs import TrainConfig
from ssd.utils import WeightsAndBiasesLogger

if __name__ == "__main__":
    coco_dir = Path("/mnt/data/datasets/object_detection/coco/")
    train_config = TrainConfig(
        num_epochs=100,
        num_classes=80,
        train_images_dir=coco_dir / "images/train2017",
        train_labels_dir=coco_dir / "labels/train2017",
        val_images_dir=coco_dir / "images/val2017",
        val_labels_dir=coco_dir / "labels/val2017",
        log_dir=Path(f"/mnt/data/code/ssd/models/{uuid4()}"),
    )

    logger = WeightsAndBiasesLogger("brrr", "ssd", f"{uuid4()}", train_config)

    for idx in range(10):
        logger.log_epoch(
            idx,
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
        )
