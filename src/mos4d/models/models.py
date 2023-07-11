#!/usr/bin/env python3
# @file      models.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved
import os
import torch
import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from mos4d.models.MinkowskiEngine.customminkunet import CustomMinkUNet
from torchmetrics import R2Score

class MOSNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.id = self.hparams["EXPERIMENT"]["ID"]
        self.lr = self.hparams["TRAIN"]["LR"]
        self.lr_epoch = hparams["TRAIN"]["LR_EPOCH"]
        self.lr_decay = hparams["TRAIN"]["LR_DECAY"]
        self.weight_decay = hparams["TRAIN"]["WEIGHT_DECAY"]
        self.loss = torch.nn.MSELoss()
        self.r2score = R2Score()
        self.model = MOSModel(hparams)

        self.data_dir = str(os.environ.get("DATA"))
        self.seqs = hparams["DATA"]["SPLIT"]["VAL"]

    def training_step(self, batch, batch_idx, dataloader_index=0):
        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, 5].reshape(-1)
        scan_indices = np.where(coordinates[:, 4].cpu().data.numpy() == 1)[0]
        scores = self.model(coordinates)
        loss = self.loss(scores[scan_indices], gt_labels[scan_indices])
        r2 = self.r2score(scores[scan_indices], gt_labels[scan_indices])
        self.log("train_loss", loss.item(), on_step=True)
        self.log("train_r2", r2.item(), on_step=True, prog_bar=True)

        torch.cuda.empty_cache()
        return {"loss": loss, "val_r2": r2}

    def validation_step(self, batch, batch_idx):
        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, 5].reshape(-1)
        scan_indices = np.where(coordinates[:, 4].cpu().data.numpy() == 1)[0]
        scores = self.model(coordinates)
        loss = self.loss(scores[scan_indices], gt_labels[scan_indices])
        r2 = self.r2score(scores[scan_indices], gt_labels[scan_indices])
        self.log("val_loss", loss.item(), on_step=True)
        self.log("val_r2", r2.item(), on_step=True, prog_bar=True)

        torch.cuda.empty_cache()
        return {"val_loss": loss, "val_r2": r2}

    def predict_step(self, batch, batch_idx):
        if batch_idx % 20 != 0:
            return
        
        for seq in self.seqs:
            coordinates = batch[:, :5].reshape(-1, 5)
            gt_labels = batch[:, 5].reshape(-1)
            scan_indices = np.where(coordinates[:, 4].cpu().data.numpy() == 1)[0]
            scores = self.model(coordinates)
            loss = self.loss(scores[scan_indices], gt_labels[scan_indices])
            r2 = self.r2score(scores[scan_indices], gt_labels[scan_indices])

            s_path = os.path.join(
                self.data_dir,
                'predictions',
                seq,
                'scans'
            )
            m_path = os.path.join(
                self.data_dir,
                'predictions',
                seq,
                'maps'
            )

            os.makedirs(s_path, exist_ok=True)
            os.makedirs(m_path, exist_ok=True)

            batch_indices = [unique.item() for unique in torch.unique(batch[:, 0])]
            for b in batch_indices:
                mask_batch = batch[:, 0] == b
                mask_scan = batch[:, -2] == 1
                mask_map  = batch[:, -2] == 0

                scan_points = batch[torch.logical_and(mask_batch, mask_scan), 1:4].cpu().data.numpy()
                scan_labels_gt = batch[torch.logical_and(mask_batch, mask_scan), -1].cpu().data.numpy()
                scan_labels_hat = scores[scan_indices].cpu().data.numpy()

                map_points = batch[torch.logical_and(mask_batch, mask_map), 1:4].cpu().data.numpy()
                map_labels_gt = batch[torch.logical_and(mask_batch, mask_map), -1].cpu().data.numpy()

                assert len(scan_points) == len(scan_labels_gt) == len(scan_labels_hat), "Lengths of arrays are not equal."

                scan_data = np.column_stack((scan_points, scan_labels_gt, scan_labels_hat))
                map_data = np.column_stack((map_points, map_labels_gt))

                scan_pth = os.path.join(s_path, str(batch_idx) + '_' + str(b) + '.npy')
                map_pth  = os.path.join(m_path, str(batch_idx) + '_' + str(b) + '.npy')
                np.save(scan_pth, scan_data)
                np.save(map_pth, map_data)

        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_epoch, gamma=self.lr_decay
        )
        return [optimizer], [scheduler]


class MOSModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ds = cfg["MODEL"]["VOXEL_SIZE"]
        self.quantization = torch.Tensor([1.0, ds, ds, ds, 1.0])
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=1, D=4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, coordinates: torch.Tensor):
        coordinates = torch.div(coordinates, self.quantization.type_as(coordinates))
        features = 0.5 * torch.ones(len(coordinates), 1).type_as(coordinates)

        tensor_field = ME.TensorField(features=features, coordinates=coordinates.type_as(features))
        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)
        out = predicted_sparse_tensor.slice(tensor_field)
        scores = self.sigmoid(out.features.reshape(-1))
        return scores
