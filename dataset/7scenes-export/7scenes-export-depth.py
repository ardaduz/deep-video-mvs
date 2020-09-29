import shutil

import cv2
import numpy as np
from path import Path

scenes = [("7scenes_chess", "01", "02"),
          ("7scenes_fire", "01", "02"),
          ("7scenes_heads", "02"),
          ("7scenes_office", "01", "03"),
          ("7scenes_pumpkin", "03", "06"),
          ("7scenes_redkitchen", "01", "07"),
          ("7scenes_stairs", "02", "06")]

input_folder = Path("/home/ardaduz/HDD/deep-mvs-dataset/raw-data/7scenes-depth")
output_folder = Path("/media/ardaduz/T5/test/7scenes")
for scene in scenes:

    if len(scene) == 3:
        folder_name, seq1, seq2 = scene
        seqs = [seq1, seq2]
    else:
        folder_name, seq1 = scene
        seqs = [seq1]

    scene_input_folder = input_folder / folder_name / "train" / "depth"

    for seq in seqs:
        files = sorted(scene_input_folder.files("seq" + seq + "*"))

        room_name = folder_name.split("_")[-1]
        scene_name = room_name + "-seq-" + seq
        scene_output_folder = output_folder / scene_name / 'depth'
        if scene_output_folder.exists():
            shutil.rmtree(scene_output_folder)
        scene_output_folder.mkdir()
        for index, file in enumerate(files):
            depth = cv2.imread(file, -1)
            depth_uint = np.round(depth).astype(np.uint16)
            save_filename = scene_output_folder / (str(index).zfill(6) + ".png")
            cv2.imwrite(save_filename, depth_uint, [cv2.IMWRITE_PNG_COMPRESSION, 3])
