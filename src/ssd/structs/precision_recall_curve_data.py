import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class PrecisionRecallCurveData(BaseModel):
    """
    Contains the precision and recall for each class at the specified confidence
    thresholds and IoU threshold.
    """

    precisions: list[Tensor] = Field(
        description=(
            "Precision values for each class at each confidence. Each list element "
            "corresponds to a different confidence. The Tensor elements correspond "
            "to the class precisions."
        ),
    )
    recalls: list[Tensor] = Field(
        description=(
            "Recall values for each class at each confidence. Each list element "
            "corresponds to a different confidence. The Tensor elements correspond "
            "to the class recalls."
        ),
    )
    confidences: list[float] = Field(
        description="The confidences the precision and recall were calculated at."
    )
    iou_threshold: float = Field(
        description="The IoU threshold the precision and recall were calculated with."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def AP(self) -> Tensor:
        """
        The per class Average Precision.
        """
        if len(self.precisions) == 0 or self.precisions[0].shape == (0,):
            raise ValueError("No precisions found.")

        if len(self.precisions) != len(self.recalls):
            msg = (
                f"Precision-recall shape mismatch: {len(self.precisions)} != "
                f"{len(self.recalls)}."
            )
            raise ValueError(msg)

        dtype = self.precisions[0].dtype
        device = self.precisions[0].device
        num_classes = self.precisions[0].shape[0]

        APs = torch.zeros((num_classes,), dtype=dtype, device=device)

        # Swap the recalls and precisions so they occur from left to right on the plot
        recalls_ltr = self.recalls[::-1]
        precisions_ltr = self.precisions[::-1]

        for idx in range(len(recalls_ltr)):
            if idx == 0:
                r0 = torch.zeros((num_classes,), dtype=dtype, device=device)
                p0 = torch.ones((num_classes,), dtype=dtype, device=device)
            else:
                r0 = recalls_ltr[idx - 1]
                p0 = precisions_ltr[idx - 1]
            r1 = recalls_ltr[idx]
            p1 = precisions_ltr[idx]

            # Calculate the mean precision betweeen r0 and r1
            delta_r = r1 - r0
            p_mean = p0 + (p1 - p0) / 2

            # Update the area under the curve
            APs += p_mean * delta_r

        return APs
