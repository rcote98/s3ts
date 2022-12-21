"""
Multitask Convolutional Classification Model

@author Raúl Coterillo
@version 2022-12
"""

# lightning
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import torch

# numpy
import numpy as np

from s3ts.network_aux import ConvEncoder, LinSeq, ConvDecoder
from s3ts.data_str import TaskParameters

import itertools
import logging

log = logging.Logger(__name__)

# ========================================================= #
#                     MULTITASK MODEL                       #
# ========================================================= #

class MultitaskModel(LightningModule):

    def __init__(self,      
        n_labels: int,
        n_patterns: int, 
        patt_length: int,
        window_size: int,
        tasks: TaskParameters,
        max_feature_maps: int = 128,
        learning_rate: float = 1e-5
        ):

        super().__init__()
        self.save_hyperparameters()

        self.n_labels = n_labels
        self.n_patterns = n_patterns
        self.patt_length = patt_length
        self.window_size = window_size
        
        self.tasks = tasks

        self.learning_rate = learning_rate

        # main encoder
        self.conv_encoder = ConvEncoder(
            in_channels=n_patterns,
            out_channels=max_feature_maps,
            conv_kernel_size=3,
            pool_kernel_size=3,
            img_height=patt_length,
            img_width=window_size
        )

        encoder_out_feats = self.conv_encoder.get_output_shape()
        
        if self.tasks.main: # main classification
            self.main_decoder = nn.Sequential(
                LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=self.n_labels,
                hid_layers=0), nn.Softmax())
        
        if self.tasks.disc: # discretized classification
            self.disc_decoder = nn.Sequential(
                LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=tasks.discrete_intervals,
                hid_layers=0), nn.Softmax())
    
        if self.tasks.pred: # discretized prediction decoder
            self.pred_decoder = nn.Sequential(
                LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=tasks.discrete_intervals,
                hid_layers=0), nn.Softmax())

        if self.tasks.areg_ts: # time series regression
            self.areg_ts_decoder = LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=self.window_size,
                hid_layers=0)

        if self.tasks.areg_img: # similarity frame regression
            self.areg_img_decoder = ConvDecoder(
                in_channels=max_feature_maps,
                out_channels=n_patterns,
                conv_kernel_size=3,
                img_height=self.conv_encoder.encoder_img_height,
                img_width=self.conv_encoder.encoder_img_width,
                encoder_feats=self.conv_encoder.encoder_feats) 

        # configure loggers
        names = [
            ("main", n_patterns), 
            ("disc", self.tasks.discrete_intervals), 
            ("pred", self.tasks.discrete_intervals)
            ]
        flags = [self.tasks.main, self.tasks.disc, self.tasks.pred]
        for (name, nclas), flag in itertools.product(names, flags):
            if flag:
                for phase in ["train", "val", "test"]:
                        self.__setattr__(f"{name}_{phase}_acc", tm.Accuracy(num_classes=nclas, task="multiclass"))
                        self.__setattr__(f"{name}_{phase}_f1",  tm.F1Score(num_classes=nclas, task="multiclass", average="micro",))
                        if phase != "train":
                            self.__setattr__(f"{name}_{phase}_auroc", tm.AUROC(num_classes=nclas, task="multiclass", average="macro"))

    # FORWARD
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def forward(self, frame):

        """ Use for inference only (separate from training_step)"""

        shared = self.conv_encoder(frame)
        results = []

        if self.tasks.main:
            main_out = self.main_decoder(shared)
            results.append(main_out)
        if self.tasks.disc:
            results.append(self.disc_decoder(shared))
        if self.tasks.pred:
            results.append(self.pred_decoder(shared))
        if self.tasks.areg_ts:
            results.append(self.areg_ts_decoder(shared))
        if self.tasks.areg_img:
            results.append(self.areg_img_decoder(shared))

        return results

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_step(self, batch, batch_idx):

        """ Prediction step, skips auxiliary tasks. """

        if self.tasks.main:
            shared = self.conv_encoder(batch)
            result = self.main_decoder(shared)
            return result
        else:
            raise NotImplementedError()

    def _inner_step(self, batch, stage: str = None):

        """ Common actions for training, test and eval steps. """

        # x[0] is the time series
        # x[1] are the sim frames
        
        x, y = batch

        results = self(x[1])
        olabel, dlabel, dlabel_pred = y

        # placeholders
        counter, outputs, losses, weights = 0, [], [], []
        if self.tasks.main:
            main_out = results[counter]
            y_true_main = F.one_hot(olabel, num_classes=self.n_labels).float()
            main_loss = F.cross_entropy(main_out, y_true_main)
            outputs.append(main_out), losses.append(main_loss), weights.append(self.tasks.main_weight)
            counter += 1
        if self.tasks.disc:
            disc_out = results[counter]
            y_true_disc = F.one_hot(dlabel, num_classes=self.tasks.discrete_intervals).float()
            disc_loss = F.cross_entropy(disc_out, y_true_disc)
            outputs.append(disc_out), losses.append(disc_loss), weights.append(self.tasks.disc_weight)
            counter += 1
        if self.tasks.pred:
            pred_out = results[counter]
            y_true_pred = F.one_hot(dlabel_pred, num_classes=self.tasks.discrete_intervals).float()
            pred_loss = F.cross_entropy(pred_out, y_true_pred)
            outputs.append(pred_out), losses.append(pred_loss), weights.append(self.tasks.pred_weight)
            counter += 1
        if self.tasks.areg_ts:
            areg_ts_out = results[counter]
            areg_ts_loss = F.mse_loss(areg_ts_out, x[0].type(torch.float32))
            outputs.append(areg_ts_out), losses.append(areg_ts_loss), weights.append(self.tasks.areg_ts_weight)
            counter += 1
        if self.tasks.areg_img:
            areg_img_out = results[counter]
            areg_img_loss = F.mse_loss(areg_img_out, x[1].type(torch.float32))
            outputs.append(areg_img_out), losses.append(areg_img_loss), weights.append(self.tasks.areg_img_weight)
            counter += 1

        # unify the loss functions in a single loss
        weights = torch.tensor(weights, dtype=torch.float32)
        losses = torch.stack(losses)
        loss = torch.exp(weights@torch.log(losses)/weights.sum())

        # accumulate and return metrics for logging
        counter = 0
        names = ["main", "disc", "pred"]
        flags = [self.tasks.main, self.tasks.disc, self.tasks.pred]
        for name, flag in zip(names, flags):
            
            if not flag: # skip if needed
                continue

            tc = counter
            acc = self.__getattr__(f"{name}_{stage}_acc")(outputs[tc], y[tc])
            f1  = self.__getattr__(f"{name}_{stage}_f1")(outputs[tc], y[tc])
            
            if stage == "train":
                self.log(f"{name}_{stage}_loss", losses[counter], sync_dist=True)
                self.log(f"{name}_{stage}_acc", acc, prog_bar=True, sync_dist=True)
                self.log(f"{name}_{stage}_f1", f1, prog_bar=True, sync_dist=True)
            else:
                self.__getattr__(f"{name}_{stage}_auroc")(outputs[tc], y[tc])

            counter += 1

        return loss.to(torch.float32)

    def training_step(self, batch, batch_idx):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch, batch_idx):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch, batch_idx):

        """ Test step. """
        return self._inner_step(batch, stage="val")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _custom_epoch_end(self, step_outputs: list[torch.Tensor], stage: str):

        """ Common actions for validation and test epoch ends. """

        if stage != "train":
            loss = torch.tensor(step_outputs).mean()
            self.log(f"{stage}_loss", loss, sync_dist=True)

        # tasks to generate metrics from
        names = ["main", "disc", "pred"]

        # metrics to analyze
        metrics = ["acc", "f1"]
        if stage != "train":
            metrics.append("auroc")

        # task flags
        flags = [self.tasks.main, self.tasks.disc, self.tasks.pred]

        print(f"\n\n  ~~ {stage} stats ~~")
        for name, flag in zip(names, flags):
            for metric in metrics:
                mstring = f"{name}_{stage}_{metric}"
                if flag:
                    val = self.__getattr__(mstring).compute()
                    if stage == "train":
                        self.log("epoch_" + mstring, val, sync_dist=True)
                    else:
                        self.log(mstring, val, sync_dist=True)
                    self.__getattr__(mstring).reset()
                    print(f"{mstring}: {val:.4f}")
        print("")

    def training_epoch_end(self, training_step_outputs):
        """ Actions to carry out at the end of each training epoch. """
        self._custom_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        """ Actions to carry out at the end of each validation epoch. """
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        """ Actions to carry out at the end of each test epoch. """
        self._custom_epoch_end(test_step_outputs, "test")

    # OPTIMIZERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def configure_optimizers(self):

        """ Define optimizers and LR schedulers. """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }