import os
import sys
import numpy as np
import cv2
from path import Path
from tqdm import tqdm

sys.path.append('.')

from utils import depth_image_to_point_cloud, write_point_cloud


def build_point_cloud(dataset_folder, scene_name, is_test=True):

    scene_folder = Path(dataset_folder) / scene_name
    poses = np.fromfile(os.path.join(scene_folder, "poses.txt"), dtype=float, sep="\n ")
    poses = np.reshape(poses, newshape=(-1, 4, 4))

    K = np.fromfile(os.path.join(scene_folder, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, newshape=(3, 3))

    if is_test:
        image_files = sorted(Path(os.path.join(scene_folder, "images")).files('*.png'))
        depth_files = sorted(Path(os.path.join(scene_folder, "depth")).files('*.png'))
        scene_points_3D = []

        counter = 1
        for i in tqdm(range(0, len(image_files), 10), desc=scene_name):
            image_file = image_files[i]
            depth_file = depth_files[i]

            rgb = cv2.imread(image_file)
            depth = cv2.imread(depth_file, -1).astype(np.float32) / 1000.0

            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=1.0, K=K, pose=poses[i])
            scene_points_3D.extend(current_points_3D)

            if counter % 30 == 0:
                part = str((counter + 1) // 30)
                write_point_cloud(scene_name + "_point_cloud_part" + part + ".ply", scene_points_3D)
                scene_points_3D.clear()

            counter = counter + 1

        write_point_cloud(scene_name + "_point_cloud_part_last.ply", scene_points_3D)


build_point_cloud("/media/ardaduz/T5/test/7scenes", "chess-seq-01", is_test=True)
