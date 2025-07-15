import wandb

from ssd.structs import TrainConfig


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
    ):
        wandb.log(
            {
                "train/class_loss": train_class_loss,
                "train/box_loss": train_box_loss,
                "val/class_loss": val_class_loss,
                "val/box_loss": val_box_loss,
                "optim/lr": learing_rate,
            },
            step=epoch,
        )
