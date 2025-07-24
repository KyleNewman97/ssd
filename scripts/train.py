from pathlib import Path

from ssd import SSD
from ssd.structs import TrainConfig

if __name__ == "__main__":
    # Define the training configuration
    coco_dir = Path("/mnt/data/datasets/object_detection/coco/")
    model_path = Path(
        "/mnt/data/code/ssd/runs/03012430-2195-4e73-b209-4aa6b731e687/last.pt"
    )
    train_config = TrainConfig(
        lr0=2e-6,
        num_epochs=100,
        num_classes=80,
        train_images_dir=coco_dir / "images/train2017",
        train_labels_dir=coco_dir / "labels/train2017",
        val_images_dir=coco_dir / "images/val2017",
        val_labels_dir=coco_dir / "labels/val2017",
        log_dir=Path("/mnt/data/code/ssd/runs/"),
        team_name="brrr",
    )

    model = SSD.load(model_path)
    model.fit(train_config)
