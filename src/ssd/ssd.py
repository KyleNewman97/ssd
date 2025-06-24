import torch

from torch import nn, Tensor
from torch.nn.functional import softmax

from ssd.anchor_box_generator import AnchorBoxGenerator
from ssd.ssd_backbone import SSDBackbone
from ssd.structs import FrameDetections


class SSD(nn.Module):
    def __init__(self, num_classes: int):
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.backbone = SSDBackbone()
        self.anchor_box_generator = AnchorBoxGenerator()

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

    def _post_process_detections(
        self,
        head_outputs: Tensor,
        anchors: Tensor,
        confidence_threshold: float,
        num_top_k: int,
    ) -> list[FrameDetections]:
        batch_size = head_outputs.shape[0]

        box_regressions = head_outputs[..., :4]
        class_probs = softmax(head_outputs[..., 4:], dim=-1)

        # Extract the "deltas" predicted by the model to the anchor boxes
        dx = box_regressions[..., 0]
        dy = box_regressions[..., 1]
        dw = box_regressions[..., 2]
        dh = box_regressions[..., 3]

        # Extract the components of the anchor boxes
        anchor_cxs = anchors[..., 0]
        anchor_cys = anchors[..., 1]
        anchor_ws = anchors[..., 2]
        anchor_hs = anchors[..., 3]

        # Calculate the predicted boxes in pixel coordinates
        pred_cx = dx * anchor_ws + anchor_cxs
        pred_cy = dy * anchor_hs + anchor_cys
        pred_w = torch.exp(dw) * anchor_ws
        pred_h = torch.exp(dh) * anchor_hs
        boxes = torch.stack((pred_cx, pred_cy, pred_w, pred_h), dim=2)

        # Find the top-k best detections for each frame/image
        frame_detections: list[FrameDetections] = []
        for im_idx in range(batch_size):
            # Extract the boxes and probabilities for the current image
            im_boxes = boxes[im_idx, ...]
            im_class_probs = class_probs[im_idx, ...]

            # Remove boxes below the specified confidence threshold
            im_max_class_probs = im_class_probs.max(dim=-1).values
            keep_box_idxs = im_max_class_probs > confidence_threshold
            kept_boxes = im_boxes[keep_box_idxs, :]
            kept_scores = im_class_probs[keep_box_idxs, :]

            # Only keep the top-k detections
            kept_max_scores, labels = kept_scores.max(dim=-1)
            top_k_scores, top_k_indices = kept_max_scores.topk(num_top_k, dim=0)
            top_k_boxes = kept_boxes.gather(
                dim=0, index=torch.stack([top_k_indices] * 4, dim=1)
            )
            top_k_labels = labels.gather(dim=0, index=top_k_indices)

            frame_detections.append(
                FrameDetections(
                    boxes=top_k_boxes, scores=top_k_scores, labels=top_k_labels
                )
            )

        return frame_detections

    def forward(self, images: Tensor, confidence_threshold: float = 0.5) -> Tensor:
        # Pass through the backbone and get feature maps from various layers
        feature_maps = self.backbone.forward(images)

        # Find the sizes of the features maps
        feature_map_sizes = [(fm.shape[2], fm.shape[3]) for fm in feature_maps]

        # Pass the feature maps through the corresponding heads
        head_outputs = [
            self.head4_3.forward(feature_maps[0]),
            self.head7.forward(feature_maps[1]),
            self.head8_2.forward(feature_maps[2]),
            self.head9_2.forward(feature_maps[3]),
            self.head10_2.forward(feature_maps[4]),
            self.head11_2.forward(feature_maps[5]),
        ]

        # Convert each head out from (N, A * K, H, W) to (N, HWA, K)
        # Afterwards concatenate all head outputs along the second dimension
        K = self.num_classes + 4
        for idx, head_output in enumerate(head_outputs):
            N, _, H, W = head_output.shape
            head_output = head_output.view(N, -1, K, H, W)
            head_output = head_output.permute(0, 3, 4, 1, 2)
            head_output = head_output.reshape(N, -1, K)
            head_outputs[idx] = head_output
        head_outputs = torch.concat(head_outputs, dim=1)

        # Create the anchor boxes
        batch_size = images.shape[0]
        image_size = (images.shape[2], images.shape[3])
        anchors = self.anchor_box_generator.forward(
            batch_size, image_size, feature_map_sizes
        )

        return head_outputs
