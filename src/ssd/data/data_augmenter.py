from torchvision.transforms import v2


class DataAugmenter(v2.Compose):
    def __init__(self, image_width: int, image_height: int):
        v2.Compose.__init__(
            self,
            [
                v2.RandomResizedCrop(
                    size=(image_height, image_width), scale=(0.3, 1), antialias=True
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ],
        )
