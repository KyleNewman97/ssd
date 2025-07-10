import torch
from torch import nn, Tensor
from torch.nn.functional import cross_entropy, smooth_l1_loss, softmax

from ssd.anchor_box_generator import AnchorBoxGenerator
from ssd.ssd_backbone import SSDBackbone
from ssd.structs import FrameDetections, FrameLabels, Losses
from ssd.utils import BoxUtils


class SSD(nn.Module):
    def __init__(self, num_classes: int):
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.backbone = SSDBackbone()
        self.anchor_box_generator = AnchorBoxGenerator()

        # Construct the heads for each of the output feature layers
        # The number of output channels is equal to:
        #   num_anchors * (num_classes + 1 + 4)
        # we add 1 above to include a background class. 4 is the number of points needed
        # to define a bounding box
        self.minor_dim_size = self.num_classes + 1 + 4
        self.head4_3 = nn.Conv2d(512, 4 * self.minor_dim_size, kernel_size=3, padding=1)
        self.head7 = nn.Conv2d(1024, 6 * self.minor_dim_size, kernel_size=3, padding=1)
        self.head8_2 = nn.Conv2d(512, 6 * self.minor_dim_size, kernel_size=3, padding=1)
        self.head9_2 = nn.Conv2d(256, 6 * self.minor_dim_size, kernel_size=3, padding=1)
        self.head10_2 = nn.Conv2d(
            256, 4 * self.minor_dim_size, kernel_size=3, padding=1
        )
        self.head11_2 = nn.Conv2d(
            256, 4 * self.minor_dim_size, kernel_size=3, padding=1
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

        # Remove the background class
        class_probs = class_probs[..., 1:]

        # Convert the regressed boxes into the image domain
        boxes = BoxUtils.regression_domain_to_image_domain(box_regressions, anchors)

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

    def compute_loss(
        self, head_outputs: Tensor, anchors: Tensor, labels: list[FrameLabels]
    ) -> Losses:
        """ """

        # Determine which anchor boxes have the highest IoU with the labels
        gt_boxes_image_domain = [l.boxes for l in labels]
        best_anchor_indices = BoxUtils.find_indices_of_best_anchor_boxes(
            anchors, gt_boxes_image_domain
        )

        # Extract the predicted boxes (in regression domain) and class logits
        boxes_regression_domain = head_outputs[..., :4]
        class_logits = head_outputs[..., 4:]

        # Loop through each image and calculate the box loss
        box_loss_list: list[Tensor] = []
        for (
            image_boxes_regression_domain,
            image_best_anchor_indices,
            image_labels,
        ) in zip(boxes_regression_domain, best_anchor_indices, labels):
            # Find the predicted boxes in the regression domain
            matched_boxes_regression_domain = image_boxes_regression_domain.gather(
                dim=1, index=image_best_anchor_indices
            )

            # Convert the ground truth boxes to the regression domain
            gt_boxes_image_domain = image_labels.boxes
            gt_boxes_regression_domain = BoxUtils.image_domain_to_regression_domain(
                gt_boxes_image_domain, anchors
            )

            # Calculate the box loss
            box_loss_list.append(
                smooth_l1_loss(
                    matched_boxes_regression_domain,
                    gt_boxes_regression_domain,
                    reduction="sum",
                )
            )
        box_loss = torch.stack(box_loss_list, dim=0)

        # Calculate the classification loss
        total_num_objects = 0
        gt_classes_list: list[Tensor] = []
        for image_class_logits, image_best_anchor_indices, image_labels in zip(
            class_logits, best_anchor_indices, labels
        ):
            # Set the label for each anchor box
            image_gt_classes = torch.zeros(
                (image_class_logits.shape[0],),
                device=self.device,
                dtype=image_class_logits.dtype,
            )
            image_gt_classes[image_best_anchor_indices] = image_labels.labels
            total_num_objects += image_labels.labels.numel()

            gt_classes_list.append(image_gt_classes)
        gt_classes = torch.stack(gt_classes_list, dim=0)

        # Calculate classification loss
        class_loss = cross_entropy(class_logits, gt_classes, reduction="none")

        N = max(1, total_num_objects)
        return Losses(box_loss=box_loss.sum() / N, class_loss=class_loss.sum() / N)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
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
        K = self.minor_dim_size
        for idx, head_output in enumerate(head_outputs):
            N, _, H, W = head_output.shape
            head_output = head_output.view(N, -1, K, H, W)
            head_output = head_output.permute(0, 3, 4, 1, 2)
            head_output = head_output.reshape(N, -1, K)
            head_outputs[idx] = head_output
        head_outputs = torch.cat(head_outputs, dim=1)

        # Create the anchor boxes
        batch_size = images.shape[0]
        image_size = (images.shape[2], images.shape[3])
        anchors = self.anchor_box_generator.forward(
            batch_size, image_size, feature_map_sizes
        )

        return head_outputs, anchors

    def infer(
        self, images: Tensor, confidence_threshold: float = 0.5, num_top_k: int = 100
    ) -> list[FrameDetections]:
        head_outputs, anchors = self.forward(images)
        return self._post_process_detections(
            head_outputs, anchors, confidence_threshold, num_top_k
        )
