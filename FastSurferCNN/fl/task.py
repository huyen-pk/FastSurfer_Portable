"""
Shared evaluation helper for Flower client/server.

Provides evaluate_dataset(trainer, dataloader, epoch=0) -> (val_loss, metrics, num_examples)
- val_loss: scalar (lower is better) suitable to return to Flower
- metrics: dict with evaluation metrics (e.g. 'miou')
- num_examples: number of examples used for evaluation
"""
from __future__ import annotations

import copy
import sys
import os
import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from yacs.config import CfgNode 

from FastSurferCNN.utils import checkpoint as cp
from FastSurferCNN.models.optimizer import get_optimizer
from FastSurferCNN.utils.lr_scheduler import get_lr_scheduler
from FastSurferCNN.utils.meters import Meter
from FastSurferCNN.train import Trainer
from FastSurferCNN.utils.misc import update_num_steps
from di.container import create_injector
from di.factories import DataLoaderFactory
from di.wrappers import TrainerBase
from fl import FederatedClientAPI

# TODO: - check configuration settings for training value syncing with server
#       - check model setting for checkpointing and load state dict
#       - check model setting for optimizer and scheduler
#       - check model setting for device

logger = logging.getLogger(__name__)
class FastSurferCNN_FL_Trainer(TrainerBase):
    def __init__(
            self, 
            trainer: Trainer, 
            data_loader_factory: DataLoaderFactory,
            federated_client: FederatedClientAPI):
        self.__trainer = trainer
        self.data_loader_factory = data_loader_factory
        self.__federated_client = federated_client

    
    def fit(self) -> Tuple[float, Dict[str, float], int, Dict[str, torch.Tensor]]:
        """
        Transfer the model to devices, create a tensor board summary writer and then perform the training loop.
        """
        trainer, cfg, model, optimizer, scheduler, device, train_loader, val_loader = self.__setup_model()
        train_meter, val_meter, start_epoch, best_miou = self.__setup_model_state(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            device=device, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            cfg=cfg
        )
        
        mean_loss, metrics, num_examples, model_state = self.__training_loop(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train_meter=train_meter,
            val_meter=val_meter,
            model=model,
            cfg=cfg,
            start_epoch=start_epoch,
            best_miou=best_miou,
        )
        metrics = {"best_miou": best_miou}
        return mean_loss, metrics, num_examples, model_state

    def __setup_model(self):
        trainer = copy.deepcopy(self.__trainer)
        cfg = trainer.cfg
        model = trainer.model
        device = trainer.device
        
        
        if cfg.NUM_GPUS > 1:
            assert (
                cfg.NUM_GPUS <= torch.cuda.device_count()
            ), "Cannot use more GPU devices than available"
            print("Using ", cfg.NUM_GPUS, "GPUs!")
            model = torch.nn.DataParallel(model)
        val_loader = self.data_loader_factory.create_dataloader(cfg, "val")
        train_loader = self.data_loader_factory.create_dataloader(cfg, "train")
        update_num_steps(train_loader, cfg)
        # Transfer the model to device(s)
        
        model = model.to(device)
        lr = cfg.TRAIN.LR
        optimizer = self.__load_optimizer(trainer, lr)
        scheduler = get_lr_scheduler(optimizer, cfg)
        return trainer, cfg, model, optimizer, scheduler, device, train_loader, val_loader

    def __load_optimizer(self, trainer: Trainer, lr: float) -> torch.optim.Optimizer:
        """
        Reset the optimizer state and set learning rate to incoming value.
        Parameters
        ----------
        trainer : Trainer
            The trainer object containing the model and optimizer to be updated.
        lr : float
            The learning rate to set for the optimizer.
        """
        optimizer = get_optimizer(trainer.model, trainer.cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return optimizer
    
    def __setup_model_state(
            self, 
            cfg: CfgNode,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            device: torch.device,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader):
        
        checkpoint_paths = cp.get_checkpoint_path(
            cfg.LOG_DIR, cfg.TRAIN.RESUME_EXPR_NUM
        )
        # Flower doesn't have api to request global state from server or peer
        # this might need to be managed by 3rd partie framework for data sharing, such as Pygrid
        # training coordination is managed by FL
        # model packaging and state management is managed by Pygrid
        # each node must either keep a copy of global state/ or be able to package their local model into a publicly available checkpoint for other nodes to access
        # or there must be another grid for data access, which may or may not contain training nodes.
        current_state = self.__federated_client.get_internal_state() 
        if current_state is not None:
            model.load_state_dict(current_state, strict=False)
        elif cfg.TRAIN.RESUME and checkpoint_paths:
            try:
                checkpoint_path = checkpoint_paths.pop()
                checkpoint_epoch, best_metric = cp.load_from_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    cfg.TRAIN.FINE_TUNE,
                )
                start_epoch = checkpoint_epoch
                best_miou = best_metric
                logger.info(f"Resume training from epoch {start_epoch}")
            except Exception as e:
                print(
                    f"No model to restore. Resuming training from Epoch 0. {e}"
                )
        else:
            logger.info("Training from scratch")
            start_epoch = 0
            best_miou = 0
        logger.info(
            f"{sum(x.numel() for x in model.parameters())} parameters in total"
        )
        # Create tensorboard summary writer
        writer = SummaryWriter(cfg.SUMMARY_PATH, flush_secs=15)
        train_meter = Meter(
            cfg,
            mode="train",
            global_step=start_epoch * len(train_loader),
            total_iter=len(train_loader),
            total_epoch=cfg.TRAIN.NUM_EPOCHS,
            device=device,
            writer=writer,
        )
        val_meter = Meter(
            cfg,
            mode="val",
            global_step=start_epoch,
            total_iter=len(val_loader),
            total_epoch=cfg.TRAIN.NUM_EPOCHS,
            device=device,
            writer=writer,
        )
        logger.info(f"Summary path {cfg.SUMMARY_PATH}")
        return train_meter, val_meter, start_epoch, best_miou

    def __training_loop(
        self,
        trainer: Trainer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_meter: Meter,
        val_meter: Meter,
        model: torch.nn.Module,
        cfg: CfgNode,
        start_epoch: int,
        best_miou: float,
        ):
        """
        Perform the training loop.
        """
        logger.info(f"Start epoch: {start_epoch + 1}")
        for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
            trainer.train(train_loader, optimizer, scheduler, train_meter, epoch=epoch)
            if epoch == cfg.TRAIN.NUM_EPOCHS - 1:
                val_meter.enable_confusion_mat()
                _, metrics, _ = self.evaluate(epoch=epoch)
                miou = metrics.get("miou", 0.0)
                val_meter.disable_confusion_mat()
            else:
                miou = trainer.eval(val_loader, val_meter, epoch=epoch)
            if (epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                logger.info(f"Saving checkpoint at epoch {epoch+1}")
                cp.save_checkpoint(
                    trainer.checkpoint_dir,
                    epoch + 1,
                    best_miou,
                    cfg.NUM_GPUS,
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                )
            if miou > best_miou:
                best_miou = miou
                logger.info(
                    f"New best checkpoint reached at epoch {epoch+1} with miou of {best_miou}\nSaving new best model."
                )
                cp.save_checkpoint(
                    trainer.checkpoint_dir,
                    epoch + 1,
                    best_miou,
                    cfg.NUM_GPUS,
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    best=True,
                )
            mean_loss = float(np.mean(val_meter.batch_losses))
            # Syncing with server
            synced = self.__federated_client.maybe_sync(model, epoch)
            if synced:
                # re-create optimizer with new global weights after sync
                optimizer = get_optimizer(model, cfg)  
                val_meter.reset()
                train_meter.reset()
        metrics = {"best_miou": best_miou}
        num_examples = self.__count_examples(train_loader)
        return mean_loss, metrics, num_examples, model.state_dict()      

    def evaluate(self, epoch: int = 0) -> Tuple[float, Dict[str, float], int]:
        """
        Evaluate the trainer.model on val_loader and return a scalar loss (lower is better),
        metrics dict and number of examples.

        Behavior:
        - Builds a Meter using trainer.cfg and the repo utilities.
        - Calls trainer.eval(val_loader, val_meter, epoch) to maintain existing logging/metrics.
        - If val_meter.batch_losses was populated by trainer.eval, use mean(batch_losses) as val_loss.
        - Otherwise, fall back to a manual pass using trainer.loss_func.
        """
        trainer = copy.deepcopy(self.__trainer)
        val_loader = self.data_loader_factory.create_dataloader(trainer.cfg, "val")
        val_meter = Meter(
            self.__trainer.cfg,
            mode="val",
            global_step=0,
            total_iter=len(val_loader) if val_loader else None,
            total_epoch=self.__trainer.cfg.TRAIN.NUM_EPOCHS,
            class_names=None,
            device=self.__trainer.device,
            writer=None,
        )

        # Try the trainer.eval path (keeps existing logging and behavior)
        try:
            eval_out = trainer.eval(val_loader, val_meter, epoch=epoch)
            # Trainer.eval returns MIoU (higher is better) in current codebase
            miou = float(eval_out) if eval_out is not None else float("nan")
        except Exception as e:
            logger.exception("trainer.eval raised exception during evaluate: %s", e)
            miou = float("nan")

        val_loss = self.__compute_eval_loss(val_loader, val_meter)
        num_examples = self.__count_examples(val_loader)

        metrics = {"miou": miou}
        return float(val_loss), metrics, int(num_examples)
        
    def __compute_eval_loss(self, trainer: Trainer, val_loader: DataLoader, val_meter: Meter) -> float:
        """Compute validation loss, preferring values from the meter."""
        if val_meter.batch_losses:
            return float(np.mean(val_meter.batch_losses))

        # fallback: manual pass computing trainer.loss_func
        try:
            loss_fn = getattr(trainer, "loss_func", None)
            if loss_fn is None:
                return float("nan")

            trainer.model.eval()
            total_loss = 0.0
            total_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(trainer.device)
                    labels = batch["label"].to(trainer.device)
                    weights = batch.get("weight", None)
                    if weights is not None:
                        weights = weights.float().to(trainer.device)
                    scale_factors = batch.get("scale_factor", None)

                    if scale_factors is not None:
                        pred = trainer.model(images, scale_factors)
                    else:
                        pred = trainer.model(images)

                    loss_total_tuple = loss_fn(pred, labels, weights)
                    if isinstance(loss_total_tuple, (list, tuple)):
                        loss_total = loss_total_tuple[0]
                    else:
                        loss_total = loss_total_tuple

                    total_loss += float(loss_total.item() if hasattr(loss_total, "item") else float(loss_total))
                    total_batches += 1
            return float(total_loss / max(1, total_batches))
        except Exception:
            logger.exception("Failed to compute validation loss in evaluate fallback")
            return float("nan")

    def __count_examples(self, data_loader: DataLoader) -> int:
        """Count the number of examples in the validation loader."""
        try:
            return len(data_loader.dataset)
        except Exception:
            try:
                # best-effort: sum batch sizes
                return sum(batch["image"].shape[0] for batch in data_loader)
            except Exception:
                return 0


