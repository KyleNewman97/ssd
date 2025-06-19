import torch
from torch import nn, Tensor
from torchvision import models


class SSDBackbone(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # SSD uses VGG16 in its backbone
        # Ensure to remove the last MaxPool layer from VGG16s feature layers
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        vgg_features: nn.Sequential = vgg.features  # type: ignore
        vgg_features = vgg_features[:-1]

        # Find the indices of the max pool layers
        maxpool_1_idx, maxpool_2_idx, maxpool_3_idx, maxpool_4_idx = [
            i for i, layer in enumerate(vgg_features) if isinstance(layer, nn.MaxPool2d)
        ]

        # In the third max pool layer ensure that it rounds up the number of cols/rows
        # so that the output dimensions match those in the paper
        maxpool_3: nn.MaxPool2d = vgg_features[maxpool_3_idx]  # type: ignore
        maxpool_3.ceil_mode = True

        # Split the VGG feature layers up into individual components
        self.conv1_2 = vgg_features[:maxpool_1_idx]
        self.conv2_2 = vgg_features[maxpool_1_idx:maxpool_2_idx]
        self.conv3_3 = vgg_features[maxpool_2_idx:maxpool_3_idx]
        self.conv4_3 = vgg_features[maxpool_3_idx:maxpool_4_idx]
        self.conv5_3 = vgg_features[maxpool_4_idx:]

        # Define the "auxilary" layers the SSD paper adds to the VGG network
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.conv11_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.to(self.device)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pass the input through the VGG feature extraction layers
        conv1_2_out = self.conv1_2(x)
        conv2_2_out = self.conv2_2(conv1_2_out)
        conv3_3_out = self.conv3_3(conv2_2_out)
        conv4_3_out = self.conv4_3(conv3_3_out)
        conv5_3_out = self.conv5_3(conv4_3_out)

        # Pass through SSDs "auxilary" layers
        conv6_out = self.conv6(conv5_3_out)
        conv7_out = self.conv7(conv6_out)
        conv8_2_out = self.conv8_2(conv7_out)
        conv9_2_out = self.conv9_2(conv8_2_out)
        conv10_2_out = self.conv10_2(conv9_2_out)
        conv11_2_out = self.conv11_2(conv10_2_out)

        return (
            conv4_3_out,
            conv7_out,
            conv8_2_out,
            conv9_2_out,
            conv10_2_out,
            conv11_2_out,
        )
