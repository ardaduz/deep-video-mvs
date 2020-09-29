import os
import shutil
from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
from path import Path


def process_scene(input_directory, output_folder):
    # For K, https://github.com/intel-isl/Open3D/issues/540
    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0, 1.0]])
    print("processing", input_directory)

    image_filenames = sorted((input_directory + "-color").files("*.jpg"))

    depth_filenames = sorted((input_directory + "-depth-clean").files("*.png"))

    pose_filename = input_directory + "-traj.txt"

    f = open(pose_filename)
    lines = f.readlines()
    f.close()

    poses = []
    for line in lines:
        elements = line.strip("\n").split(" ")
        if len(elements) < 4:
            continue
        poses.append(elements)
    poses = np.array(poses, dtype=float).reshape((-1, 4, 4))

    sequence = input_directory.split("/")[-1]
    current_output_dir = output_folder / sequence
    if os.path.isdir(current_output_dir):
        if os.path.exists("{}/poses.txt".format(current_output_dir)) and os.path.exists("{}/K.txt".format(current_output_dir)):
            return sequence
        else:
            shutil.rmtree(current_output_dir)
    os.mkdir(current_output_dir)
    os.mkdir(os.path.join(current_output_dir, "images"))
    os.mkdir(os.path.join(current_output_dir, "depth"))

    output_poses = []
    for i in range(len(poses)):
        depth_filename = depth_filenames[i]
        image_filename = image_filenames[i]

        image = cv2.imread(image_filename, -1)
        depth = cv2.imread(depth_filename, -1)
        output_poses.append(poses[i].ravel().tolist())

        cv2.imwrite("{}/images/{}.png".format(current_output_dir, str(i).zfill(6)), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite("{}/depth/{}.png".format(current_output_dir, str(i).zfill(6)), depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    output_poses = np.array(output_poses)
    np.savetxt("{}/poses.txt".format(current_output_dir), output_poses)
    np.savetxt("{}/K.txt".format(current_output_dir), K)

    return sequence


def main():
    input_folder = Path("/home/ardaduz/HDD/Downloads/iclnuim-aug-raw")
    output_folder = Path("/media/ardaduz/T5/test/iclnuim")

    input_directories = [
        input_folder / "livingroom1",
        input_folder / "livingroom2",
        input_folder / "office1",
        input_folder / "office2"]

    pool = Pool(4)
    for finished_scene in pool.imap_unordered(partial(process_scene, output_folder=output_folder), input_directories):
        print("finished", finished_scene)

    pool.join()
    pool.close()


if __name__ == '__main__':
    main()
