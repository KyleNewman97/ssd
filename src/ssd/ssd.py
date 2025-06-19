import torch

from torch import nn, Tensor

from ssd.ssd_backbone import SSDBackbone


class SSD(nn.Module):
    def __init__(self, num_classes: int):
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.backbone = SSDBackbone()

        # Construct the heads for each of the output feature layers
        # The number of output channels is equal to:
        #   num_anchors * (num_classes + 4)
        # where 4 is the number of points needed to define a bounding box
        self.head4_3 = nn.Conv2d(
            512, 4 * (self.num_classes + 4), kernel_size=3, padding=1
        )
        self.head7 = nn.Conv2d(
            1024, 6 * (self.num_classes + 4), kernel_size=3, padding=1
        )
        self.head8_2 = nn.Conv2d(
            512, 6 * (self.num_classes + 4), kernel_size=3, padding=1
        )
        self.head9_2 = nn.Conv2d(
            256, 6 * (self.num_classes + 4), kernel_size=3, padding=1
        )
        self.head10_2 = nn.Conv2d(
            256, 4 * (self.num_classes + 4), kernel_size=3, padding=1
        )
        self.head11_2 = nn.Conv2d(
            256, 4 * (self.num_classes + 4), kernel_size=3, padding=1
        )

        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        # Pass through the backbone and get feature maps from various layers
        (
            feature_map4_3,
            feature_map7,
            feature_map8_2,
            feature_map9_2,
            feature_map10_2,
            feature_map11_2,
        ) = self.backbone.forward(x)

        # Pass the feature maps through the corresponding heads
        detection_maps = [
            self.head4_3.forward(feature_map4_3),
            self.head7.forward(feature_map7),
            self.head8_2.forward(feature_map8_2),
            self.head9_2.forward(feature_map9_2),
            self.head10_2.forward(feature_map10_2),
            self.head11_2.forward(feature_map11_2),
        ]

        # Convert each detection map from (N, A * K, H, W) to (N, HWA, K)
        K = self.num_classes + 4
        for idx, detection_map in enumerate(detection_maps):
            N, _, H, W = detection_map.shape
            detection_map = detection_map.view(N, -1, K, H, W)
            detection_map = detection_map.permute(0, 3, 4, 1, 2)
            detection_map = detection_map.reshape(N, -1, K)
            detection_maps[idx] = detection_map

        return torch.concat(detection_maps, dim=1)
