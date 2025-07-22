import wandb

from ssd.structs import TrainConfig
from ssd.utils.metrics_calculator import MetricsCalculator


class WeightsAndBiasesLogger:
    def __init__(
        self, team_name: str, project: str, run_name: str, train_config: TrainConfig
    ):
        self.train_config = train_config

        wandb.init(
            entity=team_name,
            project=project,
            name=run_name,
            config=train_config.model_dump(),
        )

    def log_epoch(
        self,
        epoch: int,
        train_class_loss: float,
        train_box_loss: float,
        val_class_loss: float,
        val_box_loss: float,
        learing_rate: float,
        metrics: MetricsCalculator,
    ):
        precisions, conf, iou = metrics.summary_precisions()
        recalls, conf, iou = metrics.summary_recalls()
        mAPs = metrics.mAPs().cpu().numpy()

        precision_data = {
            f"precision/class_{i}": precisions[i] for i in range(precisions.shape[0])
        }
        recall_data = {f"recall/class_{i}": recalls[i] for i in range(recalls.shape[0])}
        mAP_data = {f"mAP@(0.5-0.95)/class_{i}": mAPs[i] for i in range(mAPs.shape[0])}

        wandb.log(
            {
                "train/class_loss": train_class_loss,
                "train/box_loss": train_box_loss,
                "val/class_loss": val_class_loss,
                "val/box_loss": val_box_loss,
                "optim/lr": learing_rate,
            }
            | precision_data
            | recall_data
            | mAP_data,
            step=epoch,
        )

    def close(self):
        wandb.finish()

    def __del__(self):
        self.close()
