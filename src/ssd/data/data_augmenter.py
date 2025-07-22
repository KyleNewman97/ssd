from torch import Tensor
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from ssd.structs import FrameLabels
from ssd.data.remove_invalid_boxes_transform import RemoveInvalidBoxesTransform


class DataAugmenter(v2.Compose):
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height

        v2.Compose.__init__(
            self,
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.25),
                v2.RandomResizedCrop(
                    size=(image_height, image_width), scale=(0.3, 1), antialias=True
                ),
                v2.ColorJitter(),
                v2.RandomGrayscale(p=0.1),
                v2.RandomApply([v2.GaussianNoise(sigma=0.04, clip=True)], p=0.2),
            ],
        )
        self.remove_invalid_boxes_transform = RemoveInvalidBoxesTransform()

    def __call__(
        self, image: Tensor, objects: FrameLabels
    ) -> tuple[Tensor, FrameLabels]:
        image, boxes = super().__call__(
            image, objects.tv_boxes(self.image_width, self.image_height)
        )

        return self.remove_invalid_boxes_transform(
            image, boxes, objects.class_ids_with_background
        )
