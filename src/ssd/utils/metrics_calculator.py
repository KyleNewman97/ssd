import torch
import numpy as np
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from ssd.structs import FrameDetections, FrameLabels
from ssd.utils.meta_logger import MetaLogger


class MetricsCalculator(MetaLogger):
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: np.ndarray | list[float] = np.linspace(0.5, 0.95, 10),
        confidence_thresholds: np.ndarray | list[float] = np.linspace(0.1, 1, 10),
    ):
        MetaLogger.__init__(self)

        self._num_classes = num_classes
        self._iou_thresholds = iou_thresholds
        self._confidence_thresholds = confidence_thresholds

        self._frame_per_class_ious: list[dict[int, Tensor]] = []
        self._frame_per_class_scores: list[dict[int, Tensor]] = []
        self._frame_per_class_label_count: list[dict[int, int]] = []

        self._tps: Tensor | None = None
        self._fps: Tensor | None = None
        self._fns: Tensor | None = None

        self._device: torch.device | None = None

    def update(
        self, frame_detections: list[FrameDetections], frame_labels: list[FrameLabels]
    ):
        # On update ensure TPs, FPs and FNs are marked as out of date
        self._tps, self._fps, self._fns = None, None, None

        for detections, labels in zip(frame_detections, frame_labels):
            # Set the device to be used
            self._device = detections.boxes.device

            detection_xyxy_boxes = box_convert(detections.boxes, "cxcywh", "xyxy")
            label_xyxy_boxes = box_convert(labels.boxes, "cxcywh", "xyxy")

            class_ids: Tensor = detections.class_ids.unique()
            self._frame_per_class_ious.append({})
            self._frame_per_class_scores.append({})
            for class_id in class_ids:
                # Calculate class masks
                det_class_mask = detections.class_ids == class_id
                lab_class_mask = labels.class_ids == class_id

                # Extract boxes and scores
                det_boxes = detection_xyxy_boxes[det_class_mask, :]
                det_scores = detections.scores[det_class_mask]
                lab_boxes = label_xyxy_boxes[lab_class_mask, :]

                # Calculate IoU for each box: N dets, M labs produces (N, M) matrix
                iou_matrix = box_iou(det_boxes, lab_boxes)

                # Add the classes box IoUs and scores
                cls_id = int(class_id.item())
                self._frame_per_class_ious[-1][cls_id] = iou_matrix
                self._frame_per_class_scores[-1][cls_id] = det_scores

            # Update the number of objects per frame per class
            label_class_ids: Tensor = labels.class_ids.unique()
            self._frame_per_class_label_count.append({})
            for class_id in label_class_ids:
                cls_id = int(class_id.item())
                self._frame_per_class_label_count[-1][cls_id] = int(
                    (labels.class_ids == class_id).sum().item()
                )

    def tps(self) -> Tensor:
        """
        Calculate the TPs on a per-class basis at various confidence and IoU thresholds.

        Returns
        -------
        tps:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's TP count.
        """
        if self._device is None:
            raise RuntimeError("Unable to infer which device to place tensors on.")

        if self._tps is not None:
            return self._tps

        num_confs = len(self._confidence_thresholds)
        num_ious = len(self._iou_thresholds)
        self._tps = torch.zeros(
            (
                num_confs,
                num_ious,
                self._num_classes,
            ),
            dtype=torch.int,
            device=self._device,
        )

        num_frames = len(self._frame_per_class_ious)
        frame_iterator = zip(self._frame_per_class_ious, self._frame_per_class_scores)
        frame_iterator = tqdm(frame_iterator, ncols=88, total=num_frames)
        # Loop over frames
        for per_class_ious, per_class_scores in frame_iterator:
            # Loop over classes observed in the frame
            for class_id in per_class_ious:
                # Find the IoU matrix and scores for specified class in this frame
                iou_matrix = per_class_ious[class_id]
                scores = per_class_scores[class_id]

                for conf_idx, conf_threshold in enumerate(self._confidence_thresholds):
                    # Set the IoU for all detections below the confidence threshold to 0
                    iou_matrix_masked = iou_matrix.clone()
                    score_mask = scores <= float(conf_threshold)
                    iou_matrix_masked[score_mask, :] = 0

                    for iou_idx, iou_threshold in enumerate(self._iou_thresholds):
                        det_idxs, lab_idxs = torch.where(
                            iou_matrix_masked >= float(iou_threshold)
                        )
                        iou_values = iou_matrix_masked[det_idxs, lab_idxs]

                        # Sort the det and lab indicies by IoU (descending)
                        sorted_idxs = torch.argsort(iou_values, descending=True)
                        det_idxs = det_idxs[sorted_idxs]
                        lab_idxs = lab_idxs[sorted_idxs]

                        # Find detection and label pairs
                        tps = 0
                        matched_dets, matched_labs = set(), set()
                        idx_iterator = zip(
                            det_idxs.cpu().tolist(), lab_idxs.cpu().tolist()
                        )
                        for det_idx, lab_idx in idx_iterator:
                            if det_idx in matched_dets or lab_idx in matched_labs:
                                continue
                            matched_dets.add(det_idx)
                            matched_labs.add(lab_idx)
                            tps += 1

                        self._tps[conf_idx, iou_idx, class_id] += tps

        return self._tps

    def fps(self) -> Tensor:
        """
        Calculate the FPs on a per-class basis at various confidence and IoU thresholds.

        Returns
        -------
        fps:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's FP count.
        """
        if self._device is None:
            raise RuntimeError("Unable to infer which device to place tensors on.")

        if self._fps is not None:
            return self._fps

        num_confs = len(self._confidence_thresholds)
        num_ious = len(self._iou_thresholds)
        num_detections = torch.zeros(
            (
                num_confs,
                num_ious,
                self._num_classes,
            ),
            dtype=torch.int,
            device=self._device,
        )

        # Calculate the number of detections
        for per_class_scores in self._frame_per_class_scores:
            for class_id in per_class_scores:
                scores = per_class_scores[class_id]
                for conf_idx, conf_threshold in enumerate(self._confidence_thresholds):
                    num_dets = (scores >= conf_threshold).sum()
                    num_detections[conf_idx, :, class_id] += num_dets

        # Calculate the number of true positives
        tps = self.tps()

        # We can calculate the number of false positives with `FPs = NUM_DETS - TPs`
        self._fps = num_detections - tps

        return self._fps

    def fns(self) -> Tensor:
        """
        Calculate the FNs on a per-class basis at various confidence and IoU thresholds.

        Returns
        -------
        fns:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's FN count.
        """
        if self._device is None:
            raise RuntimeError("Unable to infer which device to place tensors on.")

        if self._fns is not None:
            return self._fns

        num_confs = len(self._confidence_thresholds)
        num_ious = len(self._iou_thresholds)
        num_labels = torch.zeros(
            (
                num_confs,
                num_ious,
                self._num_classes,
            ),
            dtype=torch.int,
            device=self._device,
        )

        # Calculate the number of labels
        for per_class_label_count in self._frame_per_class_label_count:
            for class_id in per_class_label_count:
                count = per_class_label_count[class_id]
                num_labels[:, :, class_id] += count

        # Calculate the number of true positives
        tps = self.tps()

        # We can calculate the number of false negatives with `FNs = NUM_LABS - TPs`
        self._fns = num_labels - tps

        return self._fns

    def precisions(self) -> Tensor:
        """
        Calculate the precision on a per-class basis at various confidence and IoU
        thresholds.

        Returns
        -------
        precisions:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's precision.
        """
        precisions = self.tps() / (self.tps() + self.fps())

        # Ensure that when there are no false positives the precision equals 1
        # This is to handle the case when `tps = 0` and `fps = 0`
        precisions[self.fps() == 0] = 1

        return precisions

    def recalls(self) -> Tensor:
        """
        Calculate the recall on a per-class basis at various confidence and IoU
        thresholds.

        Returns
        -------
        recalls:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's recall.
        """
        recalls = self.tps() / (self.tps() + self.fns())

        # Ensure that when there are no false negatives the recall equals 1
        # This is to handle the case when `tps = 0` and `fns = 0`
        recalls[self.fns() == 0] = 1

        return recalls

    def APs(self) -> Tensor:
        """
        Calculate the average precision on a per-class basis at various IoU thresholds.

        Returns
        -------
        APs:
            A tensor shaped as `(num_ious, num_classes)` with each entry representing
            the corresponding `(iou, class)` pairs's average precision.
        """
        # Calculate the precision and recall - ensure that they are ordered such that
        # the recall increases with index
        precisions = self.precisions().flip(dims=(0,))
        recalls = self.recalls().flip(dims=(0,))

        # Calculate the difference between successive recall values along the PR-curve
        recall_diffs = recalls.diff(
            dim=0,
            prepend=torch.zeros(
                (1, recalls.shape[1], recalls.shape[2]),
                dtype=recalls.dtype,
                device=recalls.device,
            ),
        )

        # Find the mid point of the precision values for each recall range
        prepended_precisions = torch.ones(
            (precisions.shape[0] + 1, precisions.shape[1], precisions.shape[2]),
            dtype=precisions.dtype,
            device=precisions.device,
        )
        prepended_precisions[1:, :, :] = precisions
        precision_diffs = prepended_precisions.diff(dim=0)
        mid_precision = prepended_precisions[:-1, :, :] + precision_diffs / 2

        return (mid_precision * recall_diffs).sum(dim=0)

    def mAPs(self) -> Tensor:
        """
        Calculate the mean average precision on a per class basis.

        Returns
        -------
        APs:
            A tensor shaped as `(num_classes,)` with each entry being the mAP associated
            with the class.
        """
        APs = self.APs()
        return APs.mean(dim=0)

    def summary_precisions(self) -> tuple[np.ndarray, float, float]:
        """
        Calculates the precisions per class at the lowest IoU threshold and a mid range
        confidence threshold.
        """
        precisions = self.precisions()

        conf_idx = precisions.shape[0] // 2
        iou_idx = 0
        return (
            precisions[conf_idx, iou_idx, :].cpu().numpy(),
            self._confidence_thresholds[conf_idx],
            self._iou_thresholds[iou_idx],
        )

    def summary_recalls(self) -> tuple[np.ndarray, float, float]:
        """
        Calculates the recalls per class at the lowest IoU threshold and a mid range
        confidence threshold.
        """
        recalls = self.recalls()

        conf_idx = recalls.shape[0] // 2
        iou_idx = 0
        return (
            recalls[conf_idx, iou_idx, :].cpu().numpy(),
            self._confidence_thresholds[conf_idx],
            self._iou_thresholds[iou_idx],
        )
