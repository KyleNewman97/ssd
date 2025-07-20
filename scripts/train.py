from pathlib import Path
from uuid import uuid4

from ssd import SSD
from ssd.structs import TrainConfig

if __name__ == "__main__":
    # Define the training configuration
    coco_dir = Path("/mnt/data/datasets/object_detection/coco/")
    model_file = Path(
        "/mnt/data/code/ssd/models/23ac32e7-5881-4f99-87a5-3b556464f721/last.pt"
    )
    train_config = TrainConfig(
        lr0=2e-6,
        num_epochs=100,
        num_classes=80,
        train_images_dir=coco_dir / "images/train2017",
        train_labels_dir=coco_dir / "labels/train2017",
        val_images_dir=coco_dir / "images/val2017",
        val_labels_dir=coco_dir / "labels/val2017",
        log_dir=Path(f"/mnt/data/code/ssd/models/{uuid4()}"),
    )

    model = SSD.load(model_file)
    model.fit(train_config)
