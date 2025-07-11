from functools import partial

import torch
from torch import nn, Tensor
from torch.nn.functional import cross_entropy, smooth_l1_loss, softmax
from torch.optim import SGD
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssd.anchor_box_generator import AnchorBoxGenerator
from ssd.data import LetterboxTransform, SSDDataset
from ssd.ssd_backbone import SSDBackbone
from ssd.structs import FrameDetections, Losses, TrainConfig
from ssd.utils import BoxUtils, MetaLogger, TrainUtils


class SSD(nn.Module, MetaLogger):
    def __init__(self, num_classes: int):
        nn.Module.__init__(self)
        MetaLogger.__init__(self)

        self.num_classes = num_classes
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.backbone = SSDBackbone()
        self.anchor_box_generator = AnchorBoxGenerator(self.device)

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
        anchors = self.anchor_box_generator.forward(batch_size, feature_map_sizes)

        return head_outputs, anchors

    def fit(self, config: TrainConfig):
        # Create the data loaders for training
        train_loader, val_loader = self._create_data_loaders(config)

        # Create the optimiser and learning rate scheduler
        optimiser = SGD(
            self.parameters(),
            lr=config.lr0,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        scheduler = ChainedScheduler(
            [
                LinearLR(optimiser, total_iters=3),
                CosineAnnealingLR(optimiser, T_max=config.num_epochs),
            ],
            optimiser,
        )

        for epoch in range(config.num_epochs):
            self.logger.info(f"Epoch: {epoch}")

            # Run the training epoch
            images: Tensor
            objects: list[Tensor]
            self.train()
            tqdm_iterator = tqdm(train_loader, ncols=88)
            tqdm_iterator.set_description_str("Train")
            for idx, (images, objects) in enumerate(tqdm_iterator):
                # Zero the gradients - this is required on each mini-batch
                optimiser.zero_grad()

                # Make predictions for this batch and calculate the loss
                head_outputs, anchors = self.forward(images)
                loss = self._compute_loss(head_outputs, anchors, objects)

                # Backprop
                total_loss = loss.class_loss + loss.box_loss
                total_loss.backward()
                optimiser.step()

                tqdm_iterator.set_postfix_str(
                    f"cls_loss={loss.class_loss.item():.4}, "
                    f"box_loss={loss.box_loss.item():.4}"
                )

            # Update the learning rate scheduler
            scheduler.step()

            tqdm_iterator.close()

    def _create_data_loaders(
        self, config: TrainConfig
    ) -> tuple[DataLoader, DataLoader]:
        """
        Creates the training and validation data loaders.
        """
        # Create the transform to use with all images being fed into the model
        transform = LetterboxTransform(
            config.image_width, config.image_height, config.dtype
        )

        # Create the collate function
        collate_func = partial(TrainUtils.batch_collate_func, device=self.device)

        # Create the training dataset loader
        train_dataset = SSDDataset(
            config.train_images_dir,
            config.train_labels_dir,
            config.num_classes,
            transform,
            self.device,
            config.dtype,
        )
        train_loader = DataLoader(
            train_dataset, config.batch_size, shuffle=True, collate_fn=collate_func
        )

        # Create the validation dataset loader
        val_dataset = SSDDataset(
            config.val_images_dir,
            config.val_labels_dir,
            config.num_classes,
            transform,
            self.device,
            config.dtype,
        )
        val_loader = DataLoader(
            val_dataset, config.batch_size, shuffle=False, collate_fn=collate_func
        )

        return train_loader, val_loader

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

    def _compute_loss(
        self, head_outputs: Tensor, anchors: Tensor, objects: list[Tensor]
    ) -> Losses:
        """ """

        # Determine which anchor boxes have the highest IoU with the labels
        gt_boxes_image_domain = [o[:, 1:] for o in objects]
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
            image_objects,
            image_anchors,
        ) in zip(boxes_regression_domain, best_anchor_indices, objects, anchors):
            # Find the predicted boxes in the regression domain
            matched_boxes_regression_domain = image_boxes_regression_domain[
                image_best_anchor_indices
            ]

            # Convert the ground truth boxes to the regression domain
            gt_boxes_image_domain = image_objects[:, 1:]
            matched_anchors = image_anchors[image_best_anchor_indices]
            gt_boxes_regression_domain = BoxUtils.image_domain_to_regression_domain(
                gt_boxes_image_domain, matched_anchors
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
        for image_class_logits, image_best_anchor_indices, image_objects in zip(
            class_logits, best_anchor_indices, objects
        ):
            # Set the label for each anchor box
            image_gt_classes = torch.zeros(
                (image_class_logits.shape[0],),
                device=self.device,
                dtype=image_class_logits.dtype,
            )
            image_gt_classes[image_best_anchor_indices] = image_objects[:, 0]
            total_num_objects += image_objects[:, 0].numel()

            gt_classes_list.append(image_gt_classes)
        gt_classes = torch.stack(gt_classes_list, dim=0)

        # Calculate classification loss
        class_logits = class_logits.permute((0, 2, 1))
        gt_classes = gt_classes.to(torch.long)
        class_loss = cross_entropy(class_logits, gt_classes, reduction="none")

        N = max(1, total_num_objects)
        return Losses(box_loss=box_loss.sum() / N, class_loss=class_loss.sum() / N)

    def infer(
        self, images: Tensor, confidence_threshold: float = 0.5, num_top_k: int = 100
    ) -> list[FrameDetections]:
        head_outputs, anchors = self.forward(images)
        return self._post_process_detections(
            head_outputs, anchors, confidence_threshold, num_top_k
        )
