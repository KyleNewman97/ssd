from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

from ssd import SSD
from ssd.data import LetterboxTransform

if __name__ == "__main__":
    device = torch.device("cpu")
    model = SSD.load(
        Path("/mnt/data/code/ssd/models/91f18512-9b06-4c9a-9d2c-8330ed7458c3/best.pt"),
        device,
    )

    images_dir = Path("/mnt/data/datasets/object_detection/coco/images/val2017")
    image_files = list(images_dir.glob("*.jpg"))

    transform = LetterboxTransform()

    for file in image_files[:10]:
        detections = model.infer(file=file)

        image = pil_to_tensor(Image.open(file))
        letterbox_image = transform.transform_image(image, device)
        image = letterbox_image.permute((1, 2, 0)).numpy().astype(np.uint8)
        plt.imshow(image)

        # print(f"[{file.name}]: Num detections: {detections.boxes.shape}")
