import math
import torch

from torch import nn, Tensor


class AnchorBoxGenerator(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        nn.Module.__init__(self)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        s_min = 0.2
        s_max = 0.9
        s_range = s_max - s_min
        self.num_feature_maps = 6

        # Calculate the scales to use for each feature map
        # See page 6 of the SSD paper for further details
        self.scales = [
            s_min + s_range * (k / (self.num_feature_maps - 1))
            for k in range(self.num_feature_maps)
        ]
        self.scales.append(1.0)

        # Define the anchor box aspect ratios to use for each feature map
        self.aspect_ratios = [
            torch.tensor([1, 1, 2, 1 / 2], dtype=dtype, device=self.device),
            torch.tensor([1, 1, 2, 3, 1 / 2, 1 / 3], dtype=dtype, device=self.device),
            torch.tensor([1, 1, 2, 3, 1 / 2, 1 / 3], dtype=dtype, device=self.device),
            torch.tensor([1, 1, 2, 3, 1 / 2, 1 / 3], dtype=dtype, device=self.device),
            torch.tensor([1, 1, 2, 1 / 2], dtype=dtype, device=self.device),
            torch.tensor([1, 1, 2, 1 / 2], dtype=dtype, device=self.device),
        ]

        self._wh_pairs = self._generate_wh_pairs()

    def _generate_wh_pairs(self) -> list[Tensor]:
        """
        Calculates the (w, h) pairs of anchor boxes to use for each feature map. It
        should be noted that the size of the anchor boxes will increase as we go into
        deeper layers of the network. This is because each pixel in small receptive
        fields corresponds to a larger region in the original image.

        Returns
        -------
        wh_pairs:
            A list of tensors. Each tensor contains the normalised (relative to the base
            image) widths and heights of anchor boxes for a feature map.
        """
        wh_pairs: list[Tensor] = []

        # The notation used in the paper is also used here. Specifically:
        #   k: feature map index
        #   s_k: scale of the anchor boxes for the kth feature map
        #   s_prime_k: alternative scale for the kth feature map
        for k in range(self.num_feature_maps):
            # Calculate the cales for the kth feature map
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            scales = [s_prime_k] + [s_k for _ in range(len(self.aspect_ratios[k]) - 1)]
            scales = torch.tensor(scales, dtype=self.dtype, device=self.device)

            # Find the aspect ratios for the kth feature map
            aspect_ratios = self.aspect_ratios[k]

            # Calculate the width and height pair for each anchor box
            widths = scales * torch.sqrt(aspect_ratios)
            heights = scales / torch.sqrt(aspect_ratios)
            wh_pairs.append(torch.stack((widths, heights), dim=-1))

        return wh_pairs

    def _generate_anchor_boxes(
        self, feature_map_sizes: list[tuple[int, int]]
    ) -> Tensor:
        """
        Generates the anchor boxes for every pixel of the feature maps.

        Parameters
        ----------
        feature_map_sizes:
            The size of the feature maps (in pixels). Each element should be structured
            as `(h, w)`.

        Returns
        -------
        anchor_boxes:
            A concatenated tensor containing the anchor boxes for all feature maps.
            Each anchor box is structured as `(x, y, w, h)`.
        """
        if len(feature_map_sizes) != self.num_feature_maps:
            raise ValueError(
                f"Observed {len(feature_map_sizes)} feature map sizes, "
                f"but require {self.num_feature_maps}"
            )

        default_boxes = []

        # The same notation is used as that in the paper. Specifically:
        #   k: feature map index
        #   f_k: a tuple indicating the size of the feature map (height, width)
        for k, f_k in enumerate(feature_map_sizes):
            y_f_k, x_f_k = f_k

            # Calculate normalised offset locations for each feature map pixel
            offsets_x = (
                ((torch.arange(0, x_f_k) + 0.5) / x_f_k).to(self.dtype).to(self.device)
            )
            offsets_y = (
                ((torch.arange(0, y_f_k) + 0.5) / y_f_k).to(self.dtype).to(self.device)
            )

            # Create indicies for every distinct feature map pixel location. Ensure that
            # these are a long flat list for each axis (x and y)
            offset_idxs_y, offset_idxs_x = torch.meshgrid(
                offsets_y, offsets_x, indexing="ij"
            )
            offset_idxs_x = offset_idxs_x.reshape(-1)
            offset_idxs_y = offset_idxs_y.reshape(-1)

            # Create (x, y) pairs for every anchor box. `(offset_idxs_x, offset_idxs_y)`
            # contains distinct indicies for every pixel. We then create `x` repeats of
            # this for each anchor box. Where `x = len(self._wh_pairs[k])`.
            anchor_box_offsets = torch.stack(
                (offset_idxs_x, offset_idxs_y) * len(self._wh_pairs[k]), dim=-1
            )
            anchor_box_offsets = anchor_box_offsets.reshape(-1, 2)

            # Create the widths and heights for each anchor box
            wh_pairs = self._wh_pairs[k].repeat((y_f_k * x_f_k), 1)

            # Join the (x, y) pair with the (w, h) pair so each anchor box is defined
            # as (x, y, w, h).
            default_anchor_boxes = torch.cat((anchor_box_offsets, wh_pairs), dim=1)
            default_boxes.append(default_anchor_boxes)

        return torch.cat(default_boxes, dim=0)

    def forward(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        feature_map_sizes: list[tuple[int, int]],
    ) -> Tensor:
        """
        Generates anchor boxes for a batch of images. This assumes that all images in
        the batch have the same size.

        Parameters
        ----------
        batch_size:
            The number of images in the batch.

        image_size:
            The size of each image in the batch. This is structured as `(h, w)`.

        feature_map_sizes:
            A collection of sizes. Each size is structured as `(h, w)`. These are the
            sizes of each feature map used by the SSD head.

        Returns
        -------
        boxes:
            A tensor containing the anchor boxes for all images. The anchor boxes are
            defined as `(c_x, c_y, w, h)` in image coordinates.
        """
        wh_list = [image_size[1], image_size[0]]
        whwh = torch.tensor(wh_list + wh_list, device=self.device, dtype=self.dtype)
        anchor_boxes = self._generate_anchor_boxes(feature_map_sizes)

        boxes: list[Tensor] = []
        for _ in range(batch_size):
            boxes.append((anchor_boxes * whwh).unsqueeze(0))

        return torch.concat(boxes, dim=0)
