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

        # main classification
        if self.tasks.main:
            self.main_decoder = nn.Sequential(
                LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=self.n_labels,
                hid_layers=0), nn.Softmax())

        # discretized classification
        if self.tasks.disc:
            self.disc_decoder = nn.Sequential(
                LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=tasks.discrete_intervals,
                hid_layers=0), nn.Softmax())

        # discretized prediction decoder
        if self.tasks.pred:
            self.pred_decoder = nn.Sequential(
                LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=tasks.discrete_intervals,
                hid_layers=0), nn.Softmax())

        # time series regression
        if self.tasks.areg_ts:
            self.areg_ts_decoder = LinSeq(in_features=encoder_out_feats,
                hid_features=encoder_out_feats*2,
                out_features=self.window_size,
                hid_layers=0)

        # similarity frame regression
        if self.tasks.areg_img:
            self.areg_img_decoder = ConvDecoder(
                in_channels=max_feature_maps,
                out_channels=n_patterns,
                conv_kernel_size=3,
                img_height=self.conv_encoder.encoder_img_height,
                img_width=self.conv_encoder.encoder_img_width,
                encoder_feats=self.conv_encoder.encoder_feats)                   

        for phase in ["train", "val", "test"]:
            self.__setattr__(phase + "_acc", tm.Accuracy())
            self.__setattr__(phase + "_f1",  tm.F1Score(num_classes=n_patterns, average="micro"))
            self.__setattr__(phase + "_auroc", tm.AUROC(num_classes=n_patterns, average="macro"))

    # TODO qué pereza, innecesario
    # def get_output_shape(self):
    #     return None

    def forward(self, frame):

        """ Use for inference only (separate from training_step)"""

        shared = self.conv_encoder(frame)
        results = []

        # main task
        main_out = self.main_decoder(shared)
        results.append(main_out)

        # auxiliary tasks
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

        shared = self.conv_encoder(batch)
        result = self.main_decoder(shared)
        return result


    def _inner_step(self, x, y):

        """ Common actions for training, test and eval steps. """

        # x[0] is the time series
        # x[1] are the sim frames
        
        results = self(x[1])
        olabel, dlabel, dlabel_pred = y

        counter = 0
        losses = []
        weights = []
        
        main_out = results[0]
        y_true_main = F.one_hot(olabel, num_classes=self.n_labels).float()
        main_loss = F.cross_entropy(main_out, y_true_main)
        losses.append(main_loss)
        weights.append(self.tasks.main_weight)
        counter += 1

        if self.tasks.disc:
            disc_out = results[counter]
            y_true_disc = F.one_hot(dlabel, num_classes=self.tasks.discrete_intervals).float()
            disc_loss = F.cross_entropy(disc_out, y_true_disc)
            losses.append(disc_loss)
            weights.append(self.tasks.disc_weight)
            counter += 1

        if self.tasks.pred:
            pred_out = results[counter]
            y_true_pred = F.one_hot(dlabel_pred, num_classes=self.tasks.discrete_intervals).float()
            pred_loss = F.cross_entropy(pred_out, y_true_pred)
            losses.append(pred_loss)
            weights.append(self.tasks.pred_weight)
            counter += 1

        if self.tasks.areg_ts:
            areg_ts_out = results[counter]
            areg_ts_loss = F.mse_loss(areg_ts_out, x[0].type(torch.float32))
            losses.append(areg_ts_loss)
            weights.append(self.tasks.areg_ts_weight)
            counter += 1

        if self.tasks.areg_img:
            areg_img_out = results[counter]
            areg_img_loss = F.mse_loss(areg_img_out, x[1].type(torch.float32))
            losses.append(areg_img_loss)
            weights.append(self.tasks.areg_img_weight)
            counter += 1

        W = torch.tensor(weights, dtype=torch.float32)
        A = torch.stack(losses)
        total_loss = torch.exp(W@torch.log(A)/W.sum())

        return total_loss, main_out

    def training_step(self, batch, batch_idx):

        """ Training step. """

        self.main_task_only = False

        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        # accumulate and return metrics for logging
        acc = self.train_acc(y_pred, y[0])
        f1 = self.train_f1(y_pred, y[0])

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, sync_dist=True)

        return loss.to(torch.float32)
    
    def _custom_stats_step(self, batch, stage=None):
        
        """ Common actions for val and test step. """

        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        if stage == 'val':
            self.val_acc(y_pred, y[0])
            self.val_f1(y_pred, y[0])
            self.val_auroc(y_pred, y[0])

        elif stage == "test":
            self.test_acc(y_pred, y[0])
            self.test_f1(y_pred, y[0])
            self.test_auroc(y_pred, y[0])

        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Validation step. """
        return self._custom_stats_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """ Test step. """
        return self._custom_stats_step(batch, "test")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def training_epoch_end(self, training_step_outputs):

        """ Actions to carry out at the end of each epoch. """
        
        # compute metrics
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()

        # log metrics
        self.log("epoch_train_accuracy", train_accuracy, prog_bar=True, sync_dist=True)
        self.log("epoch_train_f1", train_f1, prog_bar=True, sync_dist=True)
        
        # reset all metrics
        self.train_acc.reset()
        self.train_f1.reset()
        print(f"\nAccuracy: {train_accuracy:.4}")
        print(f"F1 Score: {train_f1:.4}")

    def _custom_epoch_end(self, step_outputs, stage):

        """ Common actions for validation and test epoch ends. """

        if stage == "val":
            acc_metric = self.val_acc
            f1_metric = self.val_f1
            auroc_metric = self.val_auroc

        elif stage == "test":
            acc_metric = self.test_acc
            f1_metric = self.test_f1
            auroc_metric = self.test_auroc

        # compute metrics
        loss = torch.tensor(step_outputs).mean()
        accuracy = acc_metric.compute()
        f1 = f1_metric.compute()
        auroc = auroc_metric.compute()

        # log metrics
        self.log(f"{stage}_accuracy", accuracy, sync_dist=True)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"{stage}_f1", f1, sync_dist=True)
        self.log(f"{stage}_auroc", auroc, sync_dist=True)

        # reset all metrics
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        print(f"\n{stage} accuracy: {accuracy:.4} " f"f1: {f1:.4}, auroc: {auroc:.4}")

    def validation_epoch_end(self, validation_step_outputs):
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
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