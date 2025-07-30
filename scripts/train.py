from pathlib import Path

from ssd import SSD
from ssd.structs import TrainConfig

if __name__ == "__main__":
    # Define the training configuration
    coco_dir = Path("/mnt/data/datasets/object_detection/coco/")
    train_config = TrainConfig(
        lr0=1e-6,
        num_epochs=100,
        num_classes=80,
        train_images_dir=coco_dir / "images/train",
        train_labels_dir=coco_dir / "labels/train",
        val_images_dir=coco_dir / "images/val",
        val_labels_dir=coco_dir / "labels/val",
        log_dir=Path("/mnt/data/code/ssd/runs/"),
        team_name="brrr",
    )

    model = SSD(train_config.num_classes)
    model.fit(train_config)
