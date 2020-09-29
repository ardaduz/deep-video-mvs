import os
import shutil
from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
from path import Path
from scipy.spatial.transform.rotation import Rotation


def process_scene(scene_no, input_directory, output_folder):
    image_filenames = sorted((input_directory / 'imgs' / "scene_" + scene_no).files("*color*.png"))
    depth_filenames = sorted((input_directory / 'imgs' / "scene_" + scene_no).files("*depth*.png"))

    extrinsics = np.loadtxt(input_directory / 'pc' / scene_no + ".pose")
    K = np.array([[570.3, 0.0, 320.0],
                  [0.0, 570.3, 240.0],
                  [0.0, 0.0, 1.0]])

    homogen = np.zeros((1, 4))
    homogen[0, 3] = 1
    poses = []
    for extrinsic in extrinsics:
        w = extrinsic[0]
        xyz = extrinsic[1:4]
        rot = Rotation.from_quat(np.hstack((xyz, w))).as_matrix()
        tra = np.reshape(extrinsic[4:], (3, 1))
        extrinsic = np.hstack((rot, tra))
        extrinsic = np.vstack((extrinsic, homogen))
        poses.append(extrinsic)

    current_output_dir = output_folder / "scene_" + scene_no
    if os.path.isdir(current_output_dir):
        shutil.rmtree(current_output_dir)
    os.mkdir(current_output_dir)
    os.mkdir(os.path.join(current_output_dir, "images"))
    os.mkdir(os.path.join(current_output_dir, "depth"))

    output_poses = []
    for current_index in range(len(image_filenames)):
        image = cv2.imread(image_filenames[current_index])
        depth = cv2.imread(depth_filenames[current_index], cv2.IMREAD_ANYDEPTH)

        depth = np.float32(depth)
        depth = depth / 10000.0

        depth[depth > 50.0] = 0.0
        depth[np.isnan(depth)] = 0.0
        depth[np.isinf(depth)] = 0.0

        depth = (depth * 1000.0).astype(np.uint16)

        output_poses.append(poses[current_index].ravel().tolist())

        cv2.imwrite("{}/images/{}.png".format(current_output_dir, str(current_index).zfill(6)), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite("{}/depth/{}.png".format(current_output_dir, str(current_index).zfill(6)), depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    output_poses = np.array(output_poses)
    np.savetxt("{}/poses.txt".format(current_output_dir), output_poses)
    np.savetxt("{}/K.txt".format(current_output_dir), K)

    return scene_no


def main():
    output_folder = Path("/media/ardaduz/T5/test/rgbdscenes")
    input_directory = Path("/home/ardaduz/HDD/deep-mvs-dataset/raw-data/rgbd-scenes-v2-official")

    scene_nos = [str(i).zfill(2) for i in [1, 2, 5, 6, 9, 10, 13, 14]]

    pool = Pool(12)
    for finished_scene in pool.imap_unordered(partial(process_scene,
                                                      input_directory=input_directory,
                                                      output_folder=output_folder), scene_nos):
        print("finished", finished_scene)


if __name__ == '__main__':
    main()
