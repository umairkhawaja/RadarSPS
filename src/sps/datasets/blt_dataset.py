#!/usr/bin/env python3

import os
import yaml
import time
import torch
import open3d as o3d
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from scipy.spatial import cKDTree

from sps.datasets.augmentation import (
    rotate_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)

from sps.datasets import util 

#####################################################################
class BacchusModule(LightningDataModule):
    def __init__(self, cfg, test=False):
        super(BacchusModule, self).__init__()
        self.cfg = cfg
        self.test = test
        self.root_dir = str(os.environ.get("DATA"))
        self.map_path = cfg['TRAIN']['MAP']
        self.use_single_map = False
        self.maps = {}

        local_set = 'local' in self.cfg['maps_dir'] or 'local' in self.cfg['poses_dir']
        if local_set:
            raise Exception("Local coordinate system is not implemented in SPS")



        if self.test:
            print('Loading testing data ...')
            test_seqs = self.cfg['DATA']['SPLIT']['TEST']
            test_map_seqs = self.cfg['DATA']['MAPS']['TEST']
            test_scans, test_poses, test_labels, test_map_tr = self.get_scans_poses(test_seqs)
            self.test_scans = self.cash_scans(test_scans, test_poses, test_labels, test_map_tr)
            self.load_maps(test_map_seqs)

        else:
            print('Loading training data ...')
            train_seqs = self.cfg['DATA']['SPLIT']['TRAIN']
            train_map_seqs = self.cfg['DATA']['MAPS']['TRAIN']

            train_scans, train_poses, train_labels, train_map_tr = self.get_scans_poses(train_seqs)
            self.train_scans = self.cash_scans(train_scans, train_poses, train_labels, train_map_tr)

            print('Loading validating data ...')
            val_seqs = self.cfg['DATA']['SPLIT']['VAL']
            val_map_seqs = self.cfg['DATA']['MAPS']['VAL']

            val_scans, val_poses, val_labels, val_map_tr = self.get_scans_poses(val_seqs)
            self.val_scans = self.cash_scans(val_scans, val_poses, val_labels, val_map_tr)

            self.load_maps(train_map_seqs + val_map_seqs)

        # map_str = self.cfg["TRAIN"]["MAP"]
        # Load map data points, data structure: [x,y,z,label]
        # map_pth = os.path.join(self.root_dir, "maps", map_str) # If we want to use individual maps
        # map_pth = os.path.join(self.root_dir, map_str) # Just use the concatenated map of boston seaport
        # filename, file_extension = os.path.splitext(map_pth)
        # self.map = np.load(map_pth) if file_extension == '.npy' else np.loadtxt(map_pth, skiprows=1)


        if len(self.map_path):
            self.use_single_map = True

            map_ = np.loadtxt(self.map_path, skiprows=1)
            # Need to change from stability = 1 to stability = 0
            # [x y z RCS v_x v_y stable_prob]
            map_[:,-1] = 1 - map_[:,-1]
            ## Discard compensated velocities as features
            # [x y z stable_prob RCS v_x v_y]
            map_ = map_[:, [0,1,2, -1, 3,4,5]]
            self.map = map_
        else:
            all_maps = list(self.maps.values())
            self.map = np.vstack(all_maps)
        
        ## Voxel Downsample 
        # Create an Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.map[:, :3])  # Use only [x, y, z] for point cloud
        
        # Downsample the point cloud
        ref_map_sampled = point_cloud.voxel_down_sample(voxel_size=0.04)
        map_points = np.asarray(ref_map_sampled.points)
        
        # Find the closest original points to the downsampled points
        # Use KDTree for efficient nearest-neighbor search
        tree = cKDTree(self.map[:, :3])
        _, indices = tree.query(map_points)
        
        # Store the downsampled point cloud with features
        self.map = self.map[indices]

        
        ## TODO: REMOVE -- Adding for development only
        # dummy_features = np.zeros((len(self.map), 3))
        # self.map = np.hstack([self.map, dummy_features])
        # print("WARNING: USING DUMMY MAP FEATURES")

    def load_maps(self, seq_list):
        # Iterate over all files in the directory
        for seq in seq_list:
            file_path = os.path.join(self.root_dir, self.cfg['maps_dir'], f'{seq}.asc')
            map_ = np.loadtxt(file_path, skiprows=1)
            # Need to change from stability = 1 to stability = 0
            # [x y z RCS v_x v_y stable_prob]
            map_[:,-1] = 1 - map_[:,-1]
            ## Discard compensated velocities as features
            # [x y z stable_prob RCS v_x v_y]
            map_ = map_[:, [0,1,2, -1, 3,4,5]]
            self.maps[seq] = map_

    def cash_scans(self, scans_pth, poses_pth, labels_pth, map_tr_pths):
        scans_data = []
        # Zip the two lists together and iterate with tqdm
        for scan_pth, pose_pth, label_pth, map_tr_pth in tqdm(zip(scans_pth, poses_pth, labels_pth, map_tr_pths), total=len(scans_pth)):
            # Load scan and poses:
            scan_data = np.load(scan_pth)
            labels_data = np.load(label_pth)
            pose_data = np.loadtxt(pose_pth, delimiter=',')
            # Load map transformation 
            # map_transform = np.loadtxt(map_tr_pth, delimiter=',')
            map_transform = np.eye(4) # For Radar NuScenes we have aligned them using GPS poses

            # Transform the scan to the map coordinates
            # (1) First we transform the scan using the pose from SLAM system
            # (2) Second we align the transformed scan to the map using map_transform matrix
            scan_data[:,:3] = util.transform_point_cloud(scan_data[:,:3], pose_data)
            scan_data[:,:3] = util.transform_point_cloud(scan_data[:,:3], map_transform)
            
            scan_features = scan_data[:, [3,4,5]] # RCS v_x v_y
            
            scan_data = np.hstack([scan_data[:,:3], labels_data.reshape(-1, 1), scan_features])

            scans_data.append(scan_data)

        return scans_data # [scans_data[0]] # scans_data[:3]


    def get_scans_poses(self, seqs):
        seq_scans = []
        seq_poses = []
        seq_labels = []
        map_transform = []  #path to the transformation matrix that is used to align 
                            #the transformed scans (using their poses) to the base map

        for sequence in seqs:
            scans_dir = os.path.join(self.root_dir, "sequence", sequence, self.cfg['scans_dir'])
            poses_dir = os.path.join(self.root_dir, "sequence", sequence, self.cfg['poses_dir'])
            labels_dir = os.path.join(self.root_dir, "sequence", sequence, self.cfg['labels_dir'])

            scans = sorted([os.path.join(scans_dir, file) for file in os.listdir(scans_dir)], key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', f.split('.npy')[0]))))
            poses = sorted([os.path.join(poses_dir, file) for file in os.listdir(poses_dir)], key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', f.split('.txt')[0]))))
            labels = sorted([os.path.join(labels_dir, file) for file in os.listdir(labels_dir)], key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', f.split('.npy')[0]))))

            map_transform_pth = os.path.join(self.root_dir, "sequence", sequence, "map_transform")
            transform_paths = [map_transform_pth] * len(scans)

            seq_scans.extend(scans)
            seq_poses.extend(poses)
            seq_labels.extend(labels)
            map_transform.extend(transform_paths)

        assert len(seq_scans) == len(seq_poses) == len(map_transform), 'The length of those arrays should be the same!'

        return seq_scans, seq_poses, seq_labels, map_transform


    def setup(self, stage=None):
        ########## Point dataset splits
        if self.test:
            test_set = BacchusDataset(
                self.cfg, 
                self.test_scans, 
                self.map, 
                split='TEST'
            )
        else:
            train_set = BacchusDataset(
                self.cfg, 
                self.train_scans, 
                self.map, 
                split='TRAIN'
            )

            val_set = BacchusDataset(
                self.cfg, 
                self.val_scans, 
                self.map,
                split='VAL'
            )

        ########## Generate dataloaders and iterables
        if self.test:
            self.test_loader = DataLoader(
                dataset=test_set,
                batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.cfg["DATA"]["NUM_WORKER"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
            )
            self.test_iter = iter(self.test_loader)
        else:
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
    
    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):
        tensor_batch = None
        feature_tensor_batch = None
        ## MinkowskiEngine expects [x y z t b] --> 4 + 1 dims

        for i, (data) in enumerate(batch):
            if isinstance(data, tuple):
                scan_submap_data, scan_submap_features = data
            else:
                scan_submap_data = data
                scan_submap_features = None

            # Create the tensor with an additional dimension for the batch index
            ones = torch.ones(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
            tensor = torch.hstack([i * ones, scan_submap_data])
            tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])

            # Stack scan_submap_features if they exist
            if scan_submap_features is not None:
                # Create the tensor with an additional dimension for the batch index
                # ones = torch.ones(len(scan_submap_features), 1, dtype=scan_submap_features.dtype)
                # feature_tensor = torch.hstack([i * ones, scan_submap_features])
                feature_tensor = scan_submap_features
                feature_tensor_batch = feature_tensor if feature_tensor_batch is None else torch.vstack([feature_tensor_batch, feature_tensor])

        if feature_tensor_batch is not None:
            return tensor_batch, feature_tensor_batch
        else:
            return tensor_batch

#####################################################################
class BacchusDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, scans, pc_map, split):
        self.cfg = cfg
        self.scans = scans

        self.dataset_size = len(scans)

        self.map = pc_map

        # This is to find the closest map points to the scan points, this is mainly to prune the map points
        # Note: If the batch size exceeds 1, using the ME pruning function will result in an error; hence, kd_tree was employed instead.
        self.kd_tree_target = cKDTree(self.map[:,:3])

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "TRAIN"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        # Load scan and poses:
        scan_data = self.scans[idx]

        scan_points = torch.tensor(scan_data[:, :3]).to(torch.float32).reshape(-1, 3)
        scan_labels = 1 - torch.tensor(scan_data[:, 3]).to(torch.float32).reshape(-1, 1)  # convert from stable:1 to stable:0
        scan_features = torch.tensor(scan_data[:, [4,5,6]]).to(torch.float32).reshape(-1, 3)  # RCS v_x v_y
        
        # Bind time stamp to scan points
        scan_points = self.add_timestamp(scan_points, util.SCAN_TIMESTAMP)
        
        # Bind points label in the same tensor 
        scan_points = torch.hstack([scan_points, scan_labels])

        # Find the closest map to scan points (submap points), i.e. prune map points based on scan points 
        # Note: this will find the closest points that are within the voxel size.
        kd_tree_scan = cKDTree(scan_data[:, :3])
        submap_idx = self.select_closest_points(kd_tree_scan, self.kd_tree_target)
        submap_points = torch.tensor(self.map[submap_idx, :3]).to(torch.float32).reshape(-1, 3)
        submap_labels = torch.tensor(self.map[submap_idx, 3]).to(torch.float32).reshape(-1, 1)
        
        submap_features = torch.tensor(self.map[submap_idx, 4:7]).to(torch.float32).reshape(-1, 3)
        # Submap points labels are not important, so we just create a tensor of ones
        submap_labels = torch.ones(submap_points.shape[0], 1)

        # Bind time stamp to submap points
        submap_points = self.add_timestamp(submap_points, util.MAP_TIMESTAMP)

        # Bind points label in the same tensor 
        submap_points = torch.hstack([submap_points, submap_labels])

        # Bind scans and map in the same tensor 
        scan_submap_data = torch.vstack([scan_points, submap_points])
        scan_submap_features = torch.vstack([scan_features, submap_features])

        # Add plotting code
        if self.cfg.get('plotting_enabled', False):
            import matplotlib.pyplot as plt

            # Plot scan_points
            scan_points_np = scan_points[:, :3].cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.scatter(scan_points_np[:, 0], scan_points_np[:, 1], s=3, c='blue', label='Scan Points')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Scan Points Index {idx}')
            plt.legend()
            plt.savefig(f'scan_points_idx_{idx}.png')
            plt.close()


            scan_points_np = scan_points[:, :3].cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.scatter(scan_points_np[:, 0], scan_points_np[:, 1], s=3, cmap='RdYlGn', c=1-scan_labels, label='Scan Points')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Scan Points Index {idx}')
            plt.legend()
            plt.savefig(f'pred_scan_points_idx_{idx}.png')
            plt.close()

            # Plot scan+submap points together
            submap_points_np = submap_points[:, :3].cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.scatter(scan_points_np[:, 0], scan_points_np[:, 1], s=2, c='blue', label='Scan Points')
            plt.scatter(submap_points_np[:, 0], submap_points_np[:, 1], s=5, c='orange', label='Submap Points')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Scan and Submap Points Index {idx}')
            plt.legend()
            plt.savefig(f'scan_submap_points_idx_{idx}.png')
            plt.close()

        # Augment the points 
        if self.augment:
            scan_submap_data[:, :3] = self.augment_data(scan_submap_data[:, :3])

        if self.cfg['MODEL']['RADAR_FEATURES']:
            return (scan_submap_data, scan_submap_features)
        else:
            return scan_submap_data

    def add_timestamp(self, data, stamp):
        ones = torch.ones(len(data), 1, dtype=data.dtype)
        data = torch.hstack([data, ones * stamp])
        return data

    def select_points_within_radius(self, coordinates, center, radius):
        # Calculate the Euclidean distance from each point to the center
        distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
        # Select the indexes of points within the radius
        indexes = np.where(distances <= radius)[0]
        return indexes

    def select_closest_points(self, kd_tree_ref, kd_tree_target):
        start_time = time.time()

        indexes = kd_tree_ref.query_ball_tree(kd_tree_target, self.cfg["MODEL"]["VOXEL_SIZE"])
        # indexes = kd_tree_ref.query_ball_tree(kd_tree_target, 2)
        
        # Merge the indexes into one array using numpy.concatenate and list comprehension
        merged_indexes = np.concatenate([idx_list for idx_list in indexes])

        # Convert the merged_indexes to int data type
        merged_indexes = merged_indexes.astype(int)

        elapsed_time = time.time() - start_time
        
        return merged_indexes

    def augment_data(self, scan_map_batch):
        scan_map_batch = rotate_point_cloud(scan_map_batch)
        scan_map_batch = rotate_perturbation_point_cloud(scan_map_batch)
        scan_map_batch = random_flip_point_cloud(scan_map_batch)
        scan_map_batch = random_scale_point_cloud(scan_map_batch)
        return scan_map_batch

if __name__ == "__main__":
    pass
