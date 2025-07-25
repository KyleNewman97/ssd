from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, smooth_l1_loss, softmax
from torch.optim import SGD
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from ssd.anchor_box_generator import AnchorBoxGenerator
from ssd.data import DataAugmenter, LetterboxTransform, SSDDataset
from ssd.ssd_backbone import SSDBackbone
from ssd.structs import FrameDetections, FrameLabels, Losses, TrainConfig
from ssd.utils import (
    BoxUtils,
    MetaLogger,
    MetricsCalculator,
    TrainUtils,
    WeightsAndBiasesLogger,
)


class SSD(nn.Module, MetaLogger):
    def __init__(self, num_classes: int, device: torch.device | None = None):
        nn.Module.__init__(self)
        MetaLogger.__init__(self)

        self.num_classes = num_classes
        if device is None:
            self.device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        self.backbone = SSDBackbone(self.device)
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

    def set_device(self, device: torch.device):
        self.device = device
        self.backbone.set_device(device)
        self.anchor_box_generator.set_device(device)
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
        wand_logger = WeightsAndBiasesLogger(
            config.team_name, config.project_name, config.experiment_name, config
        )

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

        # Create summary writer (TensorBoard)
        self.logger.info(f"Training model: {config.experiment_name}")
        config.experiment_dir.mkdir()

        # Run over the training dataset
        best_val_loss = np.inf
        train_losses: list[Losses] = []
        num_train_losses = 100
        for epoch in range(config.num_epochs):
            self.logger.info(f"Epoch: {epoch}")

            # Run the training epoch
            images: Tensor
            objects: list[FrameLabels]
            self.train()
            tqdm_iterator = tqdm(train_loader, ncols=88)
            tqdm_iterator.set_description_str("Train")
            for images, objects in tqdm_iterator:
                # Zero the gradients - this is required on each mini-batch
                optimiser.zero_grad()

                # Make predictions for this batch and calculate the loss
                head_outputs, anchors = self.forward(images)
                loss = self._compute_loss(
                    head_outputs,
                    anchors,
                    objects,
                    config.anchor_match_iou_threshold,
                    config.anchor_background_iou_threshold,
                    config.box_loss_scaling_factor,
                )

                # Backprop
                total_loss = loss.class_loss + loss.box_loss
                total_loss.backward()
                optimiser.step()

                tqdm_iterator.set_postfix_str(
                    f"cls_loss={loss.class_loss.item():.4}, "
                    f"box_loss={loss.box_loss.item():.4}"
                )

                # Update the last `num_train_losses` stored in train_losses
                train_losses.append(loss)
                if num_train_losses < len(train_losses):
                    train_losses.pop(0)

            # Update the learning rate scheduler
            scheduler.step()

            tqdm_iterator.close()

            # Run over the validation dataset
            self.eval()
            val_losses: list[Losses] = []
            metrics = MetricsCalculator(config.num_classes)
            with torch.no_grad():
                tqdm_iterator = tqdm(val_loader, ncols=88)
                tqdm_iterator.set_description_str("Valid")
                for images, objects in tqdm_iterator:
                    head_outputs, anchors = self.forward(images)

                    # Compute loss
                    loss = self._compute_loss(
                        head_outputs,
                        anchors,
                        objects,
                        config.anchor_match_iou_threshold,
                        config.anchor_background_iou_threshold,
                        config.box_loss_scaling_factor,
                    )
                    val_losses.append(loss)

                    # Update metrics calculator
                    detections = self._post_process_detections(
                        head_outputs,
                        anchors,
                        config.min_confidence_threshold,
                        config.num_top_k,
                        config.nms_iou_threshold,
                    )
                    metrics.update(detections, objects)

                tqdm_iterator.close()

            # Calculate metrics
            t_class_loss = np.mean([loss.class_loss.item() for loss in train_losses])
            t_box_loss = np.mean([loss.box_loss.item() for loss in train_losses])
            v_class_loss = np.mean([loss.class_loss.item() for loss in val_losses])
            v_box_loss = np.mean([loss.box_loss.item() for loss in val_losses])
            val_loss = v_class_loss + v_box_loss
            self.logger.info(f"Train cls_loss={t_class_loss}, box_loss={t_box_loss}")
            self.logger.info(f"Val cls_loss={v_class_loss}, box_loss={v_box_loss}")

            # Write metrics to W&Bs
            wand_logger.log_epoch(
                epoch,
                float(t_class_loss),
                float(t_box_loss),
                float(v_class_loss),
                float(v_box_loss),
                scheduler.get_last_lr()[0],
                metrics,
            )

            # Save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info("Saving new best model.")
                self.save(config.experiment_dir / "best.pt")
            self.save(config.experiment_dir / "last.pt")

        wand_logger.close()

    def infer(
        self,
        file: Path,
        confidence_threshold: float = 0.5,
        num_top_k: int = 100,
        dtype: torch.dtype = torch.float32,
        width: int = 300,
        height: int = 300,
        nms_iou_threshold: float = 0.4,
    ) -> FrameDetections:
        # Pre-process the image
        pil_image = Image.open(file)
        image = pil_to_tensor(pil_image).to(device=self.device, dtype=dtype)
        image /= 255

        # If the image is gray scale repeat the first dimension
        if image.shape[0] == 1:
            image = image.repeat((3, 1, 1))

        # Transform to the right shape
        transform = LetterboxTransform(width, height, dtype)
        image = transform.transform_image(image, self.device)
        image = image.unsqueeze(0)

        head_outputs, anchors = self.forward(image)
        return self._post_process_detections(
            head_outputs, anchors, confidence_threshold, num_top_k, nms_iou_threshold
        )[0]

    def save(self, file: Path):
        """
        Saves the model to the specified location. By convention this location should
        end with ".pt".
        """
        torch.save(
            {
                "model": self.state_dict(),
                "num_classes": self.num_classes,
                "device": self.device.type,
            },
            file,
        )

    @classmethod
    def load(cls, file: Path, device: torch.device | None = None) -> "SSD":
        """
        Load the model from the specified location.
        """
        # Load the previous state
        model_state = torch.load(file, weights_only=True, map_location=device)
        num_classes = model_state["num_classes"]
        device = torch.device(model_state["device"]) if device is None else device

        # Initialise the model
        model = cls(num_classes, device)
        model.load_state_dict(model_state["model"])
        model.eval()

        return model

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

        # Create training data augmenter
        DataAugmenter(config.image_width, config.image_height)

        # Create the collate function
        collate_func = partial(TrainUtils.batch_collate_func, device=self.device)

        # Create the training dataset loader
        train_dataset = SSDDataset(
            config.train_images_dir,
            config.train_labels_dir,
            config.num_classes,
            transform,
            None,
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
            None,
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
        nms_iou_threshold: float,
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
            k = min(num_top_k, kept_scores.shape[0])
            kept_max_scores, labels = kept_scores.max(dim=-1)
            top_k_scores, top_k_indices = kept_max_scores.topk(k, dim=0)
            top_k_boxes = kept_boxes.gather(
                dim=0, index=torch.stack([top_k_indices] * 4, dim=1)
            )
            top_k_labels = labels.gather(dim=0, index=top_k_indices).to(torch.int)

            # Apply non-max suppression
            nms_boxes, nms_scores, nms_labels = BoxUtils.nms(
                top_k_boxes, top_k_scores, top_k_labels, nms_iou_threshold
            )

            frame_detections.append(
                FrameDetections(
                    boxes=nms_boxes, scores=nms_scores, class_ids=nms_labels
                )
            )

        return frame_detections

    def _compute_loss(
        self,
        head_outputs: Tensor,
        anchors: Tensor,
        gt_objects: list[FrameLabels],
        matching_iou_threshold: float,
        background_iou_threshold: float,
        box_loss_scaling_factor: float,
    ) -> Losses:
        """ """

        # Determine which anchor boxes have the highest IoU with the labels
        gt_boxes_image_domain = [o.boxes for o in gt_objects]
        matching_anchor_idxs, matching_gt_idxs = BoxUtils.find_anchor_gt_pairs(
            anchors, gt_boxes_image_domain, matching_iou_threshold
        )

        # Determine which anchor boxes have an IoU with GT boxes below the lower bound
        background_anchor_idxs = BoxUtils.find_anchors_meeting_iou_condition(
            anchors, gt_boxes_image_domain, background_iou_threshold, above=False
        )

        # Extract the predicted boxes (in regression domain) and class logits
        pred_boxes_regression_domain = head_outputs[..., :4]
        pred_class_logits = head_outputs[..., 4:]

        # Loop through each image and calculate the box loss
        box_loss_list: list[Tensor] = []
        for (
            image_pred_boxes_regression_domain,
            image_matching_anchor_idxs,
            image_matching_gt_idxs,
            image_gt_objects,
            image_anchors,
        ) in zip(
            pred_boxes_regression_domain,
            matching_anchor_idxs,
            matching_gt_idxs,
            gt_objects,
            anchors,
            strict=False,
        ):
            # Find the predicted boxes in the regression domain
            matched_pred_boxes_regression_domain = image_pred_boxes_regression_domain[
                image_matching_anchor_idxs, :
            ]

            # Since one ground truth box can have multiple anchor boxes associated with
            # it we may have to duplicate the GT boxes (have one for each corresponding
            # anchor box)
            gt_boxes_image_domain = image_gt_objects.boxes[image_matching_gt_idxs, :]
            matched_anchors = image_anchors[image_matching_anchor_idxs]
            gt_boxes_regression_domain = BoxUtils.image_domain_to_regression_domain(
                gt_boxes_image_domain, matched_anchors
            )

            # Calculate the box loss
            box_loss_list.append(
                smooth_l1_loss(
                    matched_pred_boxes_regression_domain,
                    gt_boxes_regression_domain,
                    reduction="sum",
                )
            )
        box_loss = torch.stack(box_loss_list, dim=0)

        # Calculate the classification loss
        total_num_objects = 0
        gt_classes_list: list[Tensor] = []
        for idx, (
            image_class_logits,
            image_matching_anchor_idxs,
            image_matching_gt_idxs,
            image_gt_objects,
            image_background_anchor_idxs,
        ) in enumerate(
            zip(
                pred_class_logits,
                matching_anchor_idxs,
                matching_gt_idxs,
                gt_objects,
                background_anchor_idxs,
                strict=False,
            )
        ):
            # Set the label for each anchor box
            num_anchors = image_class_logits.shape[0]
            image_gt_classes = torch.zeros(
                (num_anchors,), dtype=torch.int, device=self.device
            )

            # For anchors that match set the class ID - all other anchors are assumed as
            # background
            image_gt_classes[image_matching_anchor_idxs] = (
                image_gt_objects.class_ids_with_background[image_matching_gt_idxs]
            )

            # Remove background anchors that have been matched to ground truths
            mask = ~torch.isin(image_background_anchor_idxs, image_matching_anchor_idxs)
            image_background_anchor_idxs = image_background_anchor_idxs[mask]

            # Create a mask of the anchor boxes that are not associated with the
            # background or a class
            all_idxs = torch.arange(0, num_anchors, dtype=torch.int, device=self.device)
            taken_idxs = torch.cat(
                (image_matching_anchor_idxs, image_background_anchor_idxs)
            )
            not_taken_mask = ~torch.isin(all_idxs, taken_idxs)

            # Set the non background and non class anchors to 0 class in the preds
            pred_class_logits[idx, not_taken_mask, 0] = 1
            pred_class_logits[idx, not_taken_mask, 1:] = 0

            total_num_objects += image_gt_objects.class_ids_with_background.numel()

            gt_classes_list.append(image_gt_classes)
        gt_classes = torch.stack(gt_classes_list, dim=0)

        # Calculate classification loss
        pred_class_logits = pred_class_logits.permute((0, 2, 1))
        gt_classes = gt_classes.to(torch.long)
        class_loss = cross_entropy(pred_class_logits, gt_classes, reduction="none")

        N = max(1, total_num_objects)
        return Losses(
            box_loss=(box_loss.sum() / N) * box_loss_scaling_factor,
            class_loss=class_loss.sum() / N,
        )
