import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from ssd.structs import FrameDetections, FrameLabels, PrecisionRecallCurveData
from ssd.utils.meta_logger import MetaLogger


class Metrics(MetaLogger):
    def __init__(self):
        MetaLogger.__init__(self)

        self._frame_detections: list[FrameDetections] = []
        self._frame_labels: list[FrameLabels] = []

    def update(
        self, frame_detections: list[FrameDetections], frame_labels: list[FrameLabels]
    ):
        self._frame_detections += frame_detections
        self._frame_labels += frame_labels

    @staticmethod
    def frame_true_positives(
        detections: FrameDetections,
        labels: FrameLabels,
        num_classes: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> Tensor:
        """
        Calculates the number of true positives on a single frame.
        """
        class_id_to_tps = torch.zeros(
            (num_classes,), dtype=torch.int, device=detections.boxes.device
        )

        # Filter out detections below the confidence threshold
        confidence_mask = detections.scores >= confidence_threshold
        detection_boxes = detections.boxes[confidence_mask, :]
        detection_class_ids = detections.class_ids[confidence_mask]

        if detection_boxes.shape[0] == 0:
            return class_id_to_tps

        detection_xyxy_boxes = box_convert(detection_boxes, "cxcywh", "xyxy")
        label_xyxy_boxes = box_convert(labels.boxes, "cxcywh", "xyxy")

        class_ids: Tensor = detection_class_ids.unique()
        for class_id in class_ids:
            det_boxes = detection_xyxy_boxes[detection_class_ids == class_id, :]
            lab_boxes = label_xyxy_boxes[labels.class_ids == class_id, :]
            iou_matrix = box_iou(det_boxes, lab_boxes)

            # Iterate through the calculated IoUs
            # Each loop find the detection label pair with the highest IoU.
            #   - If this IoU is below the threshold then we stop - no more pairs will
            #       be above the threshold.
            #   - If the IoU is above the threshold count another true positive and then
            #       set the IoUs for the detection and label boxes to 0 so they cannot
            #       be matched again.
            # We perform this loop a maximum of `min(num_dets, num_labs)` times, because
            # after this point all values will be zero within the matrix.
            true_positives = 0
            for _ in range(min(iou_matrix.shape)):
                max_iou = iou_matrix.max()
                if max_iou < iou_threshold:
                    break

                true_positives += 1

                max_idx = iou_matrix.argmax()
                max_row, max_col = divmod(max_idx.item(), iou_matrix.shape[1])
                iou_matrix[int(max_row), :] = 0
                iou_matrix[:, int(max_col)] = 0

            class_id_to_tps[class_id.item()] = true_positives  # type: ignore

        return class_id_to_tps

    @staticmethod
    def frame_false_positives(
        detections: FrameDetections,
        labels: FrameLabels,
        num_classes: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> Tensor:
        """
        Calculates the number of false positives in a single frame.
        """
        # Filter out detections below the confidence threshold
        confidence_mask = detections.scores >= confidence_threshold
        detection_class_ids = detections.class_ids[confidence_mask]

        # Initially assume all detections are false positives
        class_id_to_fps = torch.zeros(
            (num_classes,), dtype=torch.int, device=detections.boxes.device
        )
        for class_id in range(num_classes):
            class_id_to_fps[class_id] = int(
                (detection_class_ids == class_id).sum().item()
            )

        # Calculate the true positives
        class_id_to_tps = Metrics.frame_true_positives(
            detections, labels, num_classes, confidence_threshold, iou_threshold
        )

        # Adjust the false positives based on the number of true positives observed
        for class_id in range(num_classes):
            class_id_to_fps[class_id] -= class_id_to_tps[class_id]

        return class_id_to_fps

    @staticmethod
    def frame_false_negatives(
        detections: FrameDetections,
        labels: FrameLabels,
        num_classes: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> Tensor:
        """
        Calculates the number of false negatives in a single frame.
        """
        # Initially assume all labels are false negatives
        class_id_to_fns = torch.zeros(
            (num_classes,), dtype=torch.int, device=detections.boxes.device
        )
        for class_id in range(num_classes):
            class_id_to_fns[class_id] = int((labels.class_ids == class_id).sum().item())

        # Calculate the true positives
        class_id_to_tps = Metrics.frame_true_positives(
            detections, labels, num_classes, confidence_threshold, iou_threshold
        )

        # Adjust the false negatives based on the number of true positives observed
        for class_id in range(num_classes):
            class_id_to_fns[class_id] -= class_id_to_tps[class_id]

        return class_id_to_fns

    @staticmethod
    def precision(tps: Tensor, fps: Tensor) -> Tensor:
        """
        Calculates the precision. If there are no true positives and no false positives
        this assumes perfect precision (`precision=1`) because there were no false
        positives. This is the standard behaviour for evaluating metrics on COCO.

        Parameters
        ----------
        tps:
            The true positives. This should be a tensor containing the number of true
            positives per class.

        fps:
            The false positives. This should be a tensor containing the number of false
            positives per class.

        Returns
        -------
        precision:
            The precision on a per class basis.
        """
        if tps.shape != fps.shape:
            raise ValueError(f"TPs and FPs don't match: {tps.shape} != {fps.shape}")

        tps = tps.to(torch.float32)
        fps = fps.to(torch.float32)

        precisions = tps / (tps + fps)

        # When `fps == 0` always assume perfect precision - this prevents the `nan`
        # cases when `tps = 0` and `fps = 0`
        precisions[fps == 0] = 1

        return precisions

    @staticmethod
    def recall(tps: Tensor, fns: Tensor) -> Tensor:
        """
        Calculates the recall.

        Parameters
        ----------
        tps:
            The true positives. This should be a tensor containing the number of true
            positives per class.

        fns:
            The false negatives. This should be a tensor containing the number of false
            negatives per class.

        Returns
        -------
        recall:
            The recall on a per class basis.
        """
        if tps.shape != fns.shape:
            raise ValueError(f"TPs and FNs don't match: {tps.shape} != {fns.shape}")

        tps = tps.to(torch.float32)
        fns = fns.to(torch.float32)

        recalls = tps / (tps + fns)

        # When `fns == 0` always assume perfect recall - this prevents the `nan`
        # cases when `tps = 0` and `fns = 0`, that is when this class doesn't occur in
        # the image
        recalls[fns == 0] = 1

        return recalls

    def generate_precision_recall_curve(
        self,
        num_classes: int,
        iou_threshold: float,
        confidences: list[float] = [0.05 * i for i in range(1, 21)],
    ) -> PrecisionRecallCurveData:
        """
        Calculates the precision and recall for the specified confidence thresholds
        and the IoU threshold.
        """
        if len(self._frame_detections) == 0:
            raise ValueError("Must have at least one frame of detections.")

        device = self._frame_detections[0].boxes.device

        precisions: list[Tensor] = []
        recalls: list[Tensor] = []

        # Calculate the number of TPs, FPS and FNs for each confidence
        for confidence in confidences:
            self.logger.info(f"Calculating metrics: c={confidence:.3f}")

            tps = torch.zeros((num_classes,), dtype=torch.int, device=device)
            fps = torch.zeros((num_classes,), dtype=torch.int, device=device)
            fns = torch.zeros((num_classes,), dtype=torch.int, device=device)

            tqdm_iterator = tqdm(
                zip(self._frame_detections, self._frame_labels),
                ncols=88,
                total=len(self._frame_detections),
            )
            for detections, labels in tqdm_iterator:
                tps += self.frame_true_positives(
                    detections, labels, num_classes, confidence, iou_threshold
                )
                fps += self.frame_false_positives(
                    detections, labels, num_classes, confidence, iou_threshold
                )
                fns += self.frame_false_negatives(
                    detections, labels, num_classes, confidence, iou_threshold
                )

            # Calculate the precision and recall
            precisions.append(self.precision(tps, fps))
            recalls.append(self.recall(tps, fns))

        return PrecisionRecallCurveData(
            precisions=precisions,
            recalls=recalls,
            confidences=confidences,
            iou_threshold=iou_threshold,
        )
