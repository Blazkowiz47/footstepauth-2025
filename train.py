"""
Main training file.
calls the train pipeline with configs.
"""

import argparse
import os
from os.path import join as pjoin
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torchmetrics import Accuracy
from tqdm import tqdm
import yaml
import wandb

from models import get_model
from cdatasets import get_dataset
from criterions import get_criterion
from utils import logger, set_seeds, initialise_dirs, get_run_name


parser = argparse.ArgumentParser(
    description="Training Config",
    add_help=True,
)


parser.add_argument(
    "-c",
    "--config",
    default="configs/base.yaml",
    type=str,
    help="Train config file.",
)

parser.add_argument(
    "-ckpt",
    "--continue-model",
    type=str,
    default=None,
    help="Load initial weights from partially/pretrained model.",
)

parser.add_argument(
    "--wandb",
    action="store_true",
    help="Use wandb for logging",
)

# You can add any additional arguments if you need here.


def main(config: Dict, model_name: str, args: argparse.Namespace):
    """
    Wrapper for the driver.
    """

    logfile = f"tmp/{model_name}/train.log"
    ckptdir = f"tmp/{model_name}/checkpoints"
    log = logger.get_logger(logfile, config["logger_level"])

    # Uncomment following line if you use wandb
    wandb_run_name = model_name
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="footstepauth-2025",
            name=wandb_run_name,
            config=config,
        )

    set_seeds(log, config["seed"])
    epochs = config["epochs"]
    # validate_after_epochs = config["validate_after_epochs"]

    device = "cuda"  # You can change this to cpu.

    model = get_model(config["model"], config, log)
    log.info(str(model))
    wrapper = get_dataset(config["dataset"], config, log)

    trainds = wrapper.get_split("train")
    # validationds = wrapper.get_split("validation")

    if args.continue_model:
        model.load_state_dict(torch.load(args.continue_model))

    criterion = get_criterion(config["loss"], config, log)
    metric = Accuracy(task="multiclass", num_classes=wrapper.num_classes).to(device)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    best_train_accuracy = 0
    best_mean_train_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for image, label in tqdm(trainds, desc=f"Epoch {epoch +1}"):
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)

            feats = model(image)

            step_loss, preds = criterion(feats, label)
            step_loss.backward()
            optimizer.step()

            metric.update(preds, label)
            train_losses.append(step_loss.detach().cpu().item())
            if args.wandb:
                wandb.log(
                    {
                        "train_step_loss": train_losses[-1],
                    },
                )

        train_accuracy = metric.compute()
        mean_train_loss = np.mean(train_losses)
        log.info(f"Average train step loss: {mean_train_loss}")
        log.info(f"Average train accuracy: {train_accuracy}")

        if args.wandb:
            wandb.log(
                {
                    "mean_train_accuracy": train_accuracy,
                    "mean_train_loss": mean_train_loss,
                },
                step=epoch,
            )

        metric.reset()

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(model.state_dict(), pjoin(ckptdir, "best_accuracy.pt"))

        if mean_train_loss < best_mean_train_loss:
            best_mean_train_loss = mean_train_loss
            torch.save(model.state_dict(), pjoin(ckptdir, "best_loss.pt"))

        torch.save(model.state_dict(), pjoin(ckptdir, f"epoch_{epoch}.pt"))
        if best_train_accuracy == 1.0:
            log.info("Training accuracy reached 1.0. Stopping training.")
            break

    log.info(f"Training complete. Results saved at {model_name}")


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    model_name: str = get_run_name(config["model"], config["dataset"])
    initialise_dirs(model_name)
    outputdir = f"tmp/{model_name}"
    try:
        main(config, model_name, args)
    finally:
        if args.wandb:
            wandb.finish()
        if not os.listdir(pjoin(outputdir, "checkpoints")):
            os.system(f"rm -rf  {outputdir}")
