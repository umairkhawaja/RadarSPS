#!/usr/bin/env python3
# @file      datasets.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import os
import yaml
import torch
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class BacchusModule(LightningDataModule):
    def __init__(self, cfg):
        super(BacchusModule, self).__init__()
        self.cfg = cfg
        self.root_dir = str(os.environ.get("DATA"))

        sequence = str(self.cfg["TRAIN"]["SEQUENCE"])
        scans_dir = os.path.join(self.root_dir, "sequence", sequence, "scans")
        poses_dir = os.path.join(self.root_dir, "sequence", sequence, "poses")

        self.scans = sorted([os.path.join(scans_dir, file) for file in os.listdir(scans_dir)])
        self.poses = sorted([os.path.join(poses_dir, file) for file in os.listdir(poses_dir)])

        map_str = self.cfg["TRAIN"]["MAP"]
        self.map_path = os.path.join(self.root_dir, "maps", map_str)

        self.map_transform_pth = os.path.join(self.root_dir, "sequence", sequence, "map_transform")

    def setup(self, stage=None):
        # stack the scans and poses into single numpy array column-wise
        self.files = np.column_stack((self.scans, self.poses))

        # Set a random seed for reproducibility
        np.random.seed(self.cfg["DATA"]["SEED"])

        # Get a random permutation of indices
        indices = np.random.permutation(len(self.scans))

        # split the indices into train and val a good ratio for this would be 80% and 20%
        # for now, will do the train and val, the test data will be done later
        train_indices, val_indices = np.split(indices, [int(0.8 * len(indices))])

        ########## Point dataset splits
        train_set = BacchusDataset(
            self.cfg, 
            self.files[train_indices, 0], 
            self.files[train_indices, 1],
            self.map_path, 
            self.map_transform_pth
        )

        val_set = BacchusDataset(
            self.cfg, 
            self.files[val_indices, 0], 
            self.files[val_indices, 1], 
            self.map_path,
            self.map_transform_pth
        )

        ########## Generate dataloaders and iterables
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    @staticmethod
    def collate_fn(batch):
        tensor_batch = None

        for i, (map_points, scan_points, map_labels, scan_labels) in enumerate(batch):
            ones = torch.ones(len(scan_points), 1, dtype=scan_points.dtype)
            scan_points = torch.hstack([i * ones, scan_points, 1.0 * ones, scan_labels])

            ones = torch.ones(len(map_points), 1, dtype=map_points.dtype)
            map_points = torch.hstack([i * ones, map_points, 0.0 * ones, map_labels])

            tensor = torch.vstack([scan_points, map_points])
            tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])

        return tensor_batch


class BacchusDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, scans, poses, map, map_transform_pth):
        self.cfg = cfg
        self.scans = scans
        self.poses = poses

        assert len(scans) == len(poses), "Scans [%d] and poses [%d] must have the same length" % (
            len(scans),
            len(poses),
        )

        self.dataset_size = len(scans)

        # Load map transformation 
        map_transform = np.loadtxt(map_transform_pth, delimiter=',')

        # Load map data points, data structure: [x,y,z,label]
        self.map = np.loadtxt(map)

        # Transform map to the global coordinate frames of the scans
        self.map[:,:3] = self.transform_point_cloud(self.map[:,:3], np.linalg.inv(map_transform))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        scan_pth = os.path.join(self.scans[idx])
        pose_pth = os.path.join(self.poses[idx])

        # Load scan and poses:
        scan_data = np.load(scan_pth)
        pose_data = np.loadtxt(pose_pth, delimiter=',')

        # Transform the scan to its global coordinate
        scan_data[:,:3] = self.transform_point_cloud(scan_data[:,:3], pose_data)

        # Sampling center
        center = pose_data[:3, 3]

        scan_idx = self.select_points_within_radius(scan_data[:, :3], center)
        submap_idx = self.select_points_within_radius(self.map[:, :3], center)

        scan_points = torch.tensor(scan_data[scan_idx, :3]).to(torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(scan_data[scan_idx, 3]).to(torch.float32).reshape(-1, 1)
        submap_points = torch.tensor(self.map[submap_idx, :3]).to(torch.float32).reshape(-1, 3)
        submap_labels = torch.tensor(self.map[submap_idx, 3]).to(torch.float32).reshape(-1, 1)

        return submap_points, scan_points, submap_labels, scan_labels

    def select_points_within_radius(self, coordinates, center):
        # Calculate the Euclidean distance from each point to the center
        distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
        # Select the indexes of points within the radius
        indexes = np.where(distances <= self.cfg["DATA"]["RADIUS"])[0]
        return indexes

    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Convert point cloud to homogeneous coordinates
        homogeneous_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

        # Apply transformation matrix
        transformed_coords = np.dot(homogeneous_coords, transformation_matrix.T)

        # Convert back to Cartesian coordinates
        transformed_point_cloud = transformed_coords[:, :3] / transformed_coords[:, 3][:, np.newaxis]

        return transformed_point_cloud


if __name__ == "__main__":
    # The following is mainly for testing
    config_pth = "config/config.yaml"
    cfg = yaml.safe_load(open(config_pth))

    bm = BacchusModule(cfg)
    bm.setup()

    train_dataloader = bm.train_dataloader()

    import open3d as o3d

    for batch in tqdm(train_dataloader):
        batch_indices = [unique.item() for unique in torch.unique(batch[:, 0])]
        for b in batch_indices:
            mask_batch = batch[:, 0] == b
            mask_scan = batch[:, -2] == 1

            scan_points = batch[torch.logical_and(mask_batch, mask_scan), 1:4]
            scan_labels = batch[torch.logical_and(mask_batch, mask_scan), -1]
            map_points = batch[torch.logical_and(mask_batch, ~mask_scan), 1:4]
            map_labels = batch[torch.logical_and(mask_batch, ~mask_scan), -1]

            # Scan
            pcd_scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_points.numpy()))
            scan_colors = np.zeros((len(scan_labels), 3))
            scan_colors[:, 0] = 0.5
            scan_colors[:, 0] = 0.5 * (1 + scan_labels.numpy())
            pcd_scan.colors = o3d.utility.Vector3dVector(scan_colors)

            # Map
            pcd_map = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_points.numpy()))
            map_colors = np.zeros((len(map_labels), 3))
            map_colors[:, 1] = 0.5
            map_colors[:, 1] = 0.5 * (1 + map_labels.numpy())
            pcd_map.colors = o3d.utility.Vector3dVector(map_colors)

            o3d.visualization.draw_geometries([pcd_scan, pcd_map])
