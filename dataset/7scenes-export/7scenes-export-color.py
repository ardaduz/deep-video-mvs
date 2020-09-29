import os
import shutil
from multiprocessing.pool import Pool

import cv2
import numpy as np
from functools import partial
from path import Path


def process_scene(input_directory, output_folder):
    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0, 1.0]])
    print("processing", input_directory)
    image_filenames = sorted(input_directory.files("*color.png"))
    pose_filenames = sorted(input_directory.files("*pose.txt"))

    poses = []
    for pose_filename in pose_filenames:
        pose = np.loadtxt(pose_filename)
        poses.append(pose)

    scene = input_directory.split("/")[-2]
    seq = input_directory.split("/")[-1]
    current_output_dir = output_folder / scene + "-" + seq
    if os.path.isdir(current_output_dir):
        if os.path.exists("{}/poses.txt".format(current_output_dir)) and os.path.exists("{}/K.txt".format(current_output_dir)):
            return scene
        else:
            shutil.rmtree(current_output_dir)

    os.mkdir(current_output_dir)
    os.mkdir(os.path.join(current_output_dir, "images"))

    output_poses = []
    for current_index in range(len(image_filenames)):
        image = cv2.imread(image_filenames[current_index])

        output_poses.append(poses[current_index].ravel().tolist())

        cv2.imwrite("{}/images/{}.png".format(current_output_dir, str(current_index).zfill(6)), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    output_poses = np.array(output_poses)
    np.savetxt("{}/poses.txt".format(current_output_dir), output_poses)
    np.savetxt("{}/K.txt".format(current_output_dir), K)

    return scene


def main():
    input_folder = Path("/home/ardaduz/HDD/deep-mvs-dataset/raw-data/7scenes-official")
    output_folder = Path("/media/ardaduz/T5/test/7scenes")

    input_directories = [
        input_folder / "redkitchen/seq-01",
        input_folder / "redkitchen/seq-07",
        input_folder / "chess/seq-01",
        input_folder / "chess/seq-02",
        input_folder / "heads/seq-02",
        input_folder / "fire/seq-01",
        input_folder / "fire/seq-02",
        input_folder / "office/seq-01",
        input_folder / "office/seq-03",
        input_folder / "pumpkin/seq-03",
        input_folder / "pumpkin/seq-06",
        input_folder / "stairs/seq-02",
        input_folder / "stairs/seq-06"]

    pool = Pool(6)
    for finished_scene in pool.imap_unordered(partial(process_scene, output_folder=output_folder), input_directories):
        print("finished", finished_scene)

    pool.join()
    pool.close()


if __name__ == '__main__':
    main()
