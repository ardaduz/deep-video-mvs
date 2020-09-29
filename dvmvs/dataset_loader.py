import copy
import random
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
from kornia import adjust_brightness, adjust_gamma, adjust_contrast
from path import Path
from torch.utils.data import Dataset, DataLoader

from dvmvs.config import Config
from dvmvs.utils import pose_distance


def is_valid_pair(reference_pose, measurement_pose, pose_dist_min, pose_dist_max, t_norm_threshold=0.05, return_measure=False):
    combined_measure, R_measure, t_measure = pose_distance(reference_pose, measurement_pose)

    if pose_dist_min <= combined_measure <= pose_dist_max and t_measure >= t_norm_threshold:
        result = True
    else:
        result = False

    if return_measure:
        return result, combined_measure
    else:
        return result


def gather_pairs_train(poses, used_pairs, is_backward, initial_pose_dist_min, initial_pose_dist_max):
    sequence_length = len(poses)
    while_range = range(0, sequence_length)

    pose_dist_min = copy.deepcopy(initial_pose_dist_min)
    pose_dist_max = copy.deepcopy(initial_pose_dist_max)

    used_measurement_indices = set()

    # Gather pairs
    check_future = False
    pairs = []

    if is_backward:
        i = sequence_length - 1
        step = -1
        first_limit = 5
        second_limit = sequence_length - 5
    else:
        i = 0
        step = 1
        first_limit = sequence_length - 5
        second_limit = 5

    loosening_counter = 0
    while i in while_range:
        pair = (i, -1)

        if check_future:
            for j in range(i + step, first_limit, step):
                if j not in used_measurement_indices and (i, j) not in used_pairs:
                    valid = is_valid_pair(poses[i], poses[j], pose_dist_min, pose_dist_max)

                    if valid:
                        pair = (i, j)
                        pairs.append(pair)
                        used_pairs.add(pair)
                        used_pairs.add((pair[1], pair[0]))
                        used_measurement_indices.add(j)
                        pose_dist_min = copy.deepcopy(initial_pose_dist_min)
                        pose_dist_max = copy.deepcopy(initial_pose_dist_max)
                        i += step
                        check_future = False
                        loosening_counter = 0
                        break
        else:
            for j in range(i - step, second_limit, -step):
                if j not in used_measurement_indices and (i, j) not in used_pairs:
                    valid = is_valid_pair(poses[i], poses[j], pose_dist_min, pose_dist_max)

                    if valid:
                        pair = (i, j)
                        pairs.append(pair)
                        used_pairs.add(pair)
                        used_pairs.add((pair[1], pair[0]))
                        used_measurement_indices.add(j)
                        pose_dist_min = copy.deepcopy(initial_pose_dist_min)
                        pose_dist_max = copy.deepcopy(initial_pose_dist_max)
                        i += step
                        check_future = False
                        loosening_counter = 0
                        break

        if pair[1] == -1:
            if check_future:
                pose_dist_min = pose_dist_min / 1.1
                pose_dist_max = pose_dist_max * 1.1
                check_future = False
                loosening_counter += 1
                if loosening_counter > 1:
                    i += step
                    loosening_counter = 0
            else:
                check_future = True
        else:
            check_future = False

    return pairs


def crawl_subprocess_short(scene, dataset_path, count, progress):
    scene_path = Path(dataset_path) / scene
    poses = np.reshape(np.loadtxt(scene_path / "poses.txt"), newshape=(-1, 4, 4))

    samples = []
    used_pairs = set()

    for multiplier in [(1.0, False), (0.666, True), (1.5, False)]:
        pairs = gather_pairs_train(poses, used_pairs,
                                   is_backward=multiplier[1],
                                   initial_pose_dist_min=multiplier[0] * Config.train_minimum_pose_distance,
                                   initial_pose_dist_max=multiplier[0] * Config.train_maximum_pose_distance)

        for pair in pairs:
            i, j = pair
            sample = {'scene': scene,
                      'indices': [i, j]}
            samples.append(sample)

    progress.value += 1
    print(progress.value, "/", count, end='\r')

    return samples


def crawl_subprocess_long(scene, dataset_path, count, progress, subsequence_length):
    scene_path = Path(dataset_path) / scene
    poses = np.reshape(np.loadtxt(scene_path / "poses.txt"), newshape=(-1, 4, 4))
    sequence_length = np.shape(poses)[0]

    used_pairs = set()

    usage_threshold = 1
    used_nodes = dict()
    for i in range(sequence_length):
        used_nodes[i] = 0

    calculated_step = Config.train_crawl_step
    samples = []
    for offset, multiplier, is_backward in [(0 % calculated_step, 1.0, False),
                                            (1 % calculated_step, 0.666, True),
                                            (2 % calculated_step, 1.5, False),
                                            (3 % calculated_step, 0.8, True),
                                            (4 % calculated_step, 1.25, False),
                                            (5 % calculated_step, 1.0, True),
                                            (6 % calculated_step, 0.666, False),
                                            (7 % calculated_step, 1.5, True),
                                            (8 % calculated_step, 0.8, False),
                                            (9 % calculated_step, 1.25, True)]:

        if is_backward:
            start = sequence_length - 1 - offset
            step = -calculated_step
            limit = subsequence_length
        else:
            start = offset
            step = calculated_step
            limit = sequence_length - subsequence_length + 1

        for i in range(start, limit, step):
            if used_nodes[i] > usage_threshold:
                continue

            sample = {'scene': scene,
                      'indices': [i]}

            previous_index = i
            valid_counter = 1
            any_counter = 1
            reached_sequence_limit = False
            while valid_counter < subsequence_length:

                if is_backward:
                    j = i - any_counter
                    reached_sequence_limit = j < 0
                else:
                    j = i + any_counter
                    reached_sequence_limit = j >= sequence_length

                if not reached_sequence_limit:
                    current_index = j

                    check1 = used_nodes[current_index] <= usage_threshold
                    check2 = (previous_index, current_index) not in used_pairs
                    check3 = is_valid_pair(poses[previous_index],
                                           poses[current_index],
                                           multiplier * Config.train_minimum_pose_distance,
                                           multiplier * Config.train_maximum_pose_distance,
                                           t_norm_threshold=multiplier * Config.train_minimum_pose_distance * 0.5)

                    if check1 and check2 and check3:
                        sample['indices'].append(current_index)
                        previous_index = copy.deepcopy(current_index)
                        valid_counter += 1
                    any_counter += 1
                else:
                    break

            if not reached_sequence_limit:
                previous_node = sample['indices'][0]
                used_nodes[previous_node] += 1
                for current_node in sample['indices'][1:]:
                    used_nodes[current_node] += 1
                    used_pairs.add((previous_node, current_node))
                    used_pairs.add((current_node, previous_node))
                    previous_node = copy.deepcopy(current_node)

                samples.append(sample)

    progress.value += 1
    print(progress.value, "/", count, end='\r')
    return samples


def crawl(dataset_path, scenes, subsequence_length, num_workers=1):
    pool = Pool(num_workers)
    manager = Manager()

    count = len(scenes)
    progress = manager.Value('i', 0)

    samples = []

    if subsequence_length == 2:
        for scene_samples in pool.imap_unordered(partial(crawl_subprocess_short,
                                                         dataset_path=dataset_path,
                                                         count=count,
                                                         progress=progress), scenes):
            samples.extend(scene_samples)

    else:
        for scene_samples in pool.imap_unordered(partial(crawl_subprocess_long,
                                                         dataset_path=dataset_path,
                                                         count=count,
                                                         progress=progress,
                                                         subsequence_length=subsequence_length), scenes):
            samples.extend(scene_samples)

    random.shuffle(samples)

    return samples


def read_split(path):
    scenes_txt = np.loadtxt(path, dtype=str, delimiter="\n")
    return scenes_txt


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth(path, scaling=1000.0):
    depth = np.load(path).astype(np.float32) / scaling
    return depth


class PreprocessImage:
    def __init__(self, K, old_width, old_height, new_width, new_height, distortion_crop=0, perform_crop=True):
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.new_width = new_width
        self.new_height = new_height
        self.perform_crop = perform_crop

        original_height = np.copy(old_height)
        original_width = np.copy(old_width)

        if self.perform_crop:
            old_height -= 2 * distortion_crop
            old_width -= 2 * distortion_crop

            old_aspect_ratio = float(old_width) / float(old_height)
            new_aspect_ratio = float(new_width) / float(new_height)

            if old_aspect_ratio > new_aspect_ratio:
                # we should crop horizontally to decrease image width
                target_width = old_height * new_aspect_ratio
                self.crop_x = int(np.floor((old_width - target_width) / 2.0)) + distortion_crop
                self.crop_y = distortion_crop
            else:
                # we should crop vertically to decrease image height
                target_height = old_width / new_aspect_ratio
                self.crop_x = distortion_crop
                self.crop_y = int(np.floor((old_height - target_height) / 2.0)) + distortion_crop

            self.cx -= self.crop_x
            self.cy -= self.crop_y
            intermediate_height = original_height - 2 * self.crop_y
            intermediate_width = original_width - 2 * self.crop_x

            factor_x = float(new_width) / float(intermediate_width)
            factor_y = float(new_height) / float(intermediate_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y
        else:
            self.crop_x = 0
            self.crop_y = 0
            factor_x = float(new_width) / float(original_width)
            factor_y = float(new_height) / float(original_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y

    def apply_depth(self, depth):
        raw_height, raw_width = depth.shape
        cropped_depth = depth[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x]
        resized_cropped_depth = cv2.resize(cropped_depth, (self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)
        return resized_cropped_depth

    def apply_rgb(self, image, scale_rgb, mean_rgb, std_rgb, normalize_colors=True):
        raw_height, raw_width, _ = image.shape
        cropped_image = image[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x, :]
        cropped_image = cv2.resize(cropped_image, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        if normalize_colors:
            cropped_image = cropped_image / scale_rgb
            cropped_image[:, :, 0] = (cropped_image[:, :, 0] - mean_rgb[0]) / std_rgb[0]
            cropped_image[:, :, 1] = (cropped_image[:, :, 1] - mean_rgb[1]) / std_rgb[1]
            cropped_image[:, :, 2] = (cropped_image[:, :, 2] - mean_rgb[2]) / std_rgb[2]
        return cropped_image

    def get_updated_intrinsics(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])


class MVSDataset(Dataset):
    def __init__(self, root, seed, split, subsequence_length, scale_rgb, mean_rgb, std_rgb, geometric_scale_augmentation=False):
        np.random.seed(seed)
        random.seed(seed)

        self.subsequence_length = subsequence_length
        self.geometric_scale_augmentation = geometric_scale_augmentation
        self.root = Path(root)
        self.split = split

        if split == "TRAINING":
            self.scenes = read_split(self.root / "train.txt")
        elif split == "VALIDATION":
            self.scenes = read_split(self.root / "validation.txt")

        # self.scenes = self.scenes[0:20]
        self.samples = crawl(dataset_path=self.root,
                             scenes=self.scenes,
                             subsequence_length=self.subsequence_length,
                             num_workers=Config.train_data_pipeline_workers)

        self.scale_rgb = scale_rgb
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb

    def __getitem__(self, sample_index):
        sample = self.samples[sample_index]

        scene = sample['scene']
        indices = sample['indices']

        scene_path = self.root / scene

        K = np.loadtxt(scene_path / 'K.txt', dtype=np.float32)
        scene_poses = np.reshape(np.loadtxt(scene_path / 'poses.txt', dtype=np.float32), newshape=(-1, 4, 4))
        scene_npzs = sorted(scene_path.files('*.npz'))

        if self.split == "TRAINING" and np.random.random() > 0.5:
            indices.reverse()

        raw_poses = []
        raw_images = []
        raw_depths = []
        for i in indices:
            data = np.load(scene_npzs[i])
            raw_images.append(data['image'])
            raw_depths.append(data['depth'])
            raw_poses.append(scene_poses[i])

        preprocessor = PreprocessImage(K=K,
                                       old_width=raw_images[0].shape[1],
                                       old_height=raw_depths[0].shape[0],
                                       new_width=Config.train_image_width,
                                       new_height=Config.train_image_height,
                                       distortion_crop=0)

        output_images = []
        output_depths = []
        output_poses = []

        rgb_sum = 0
        min_depth_in_sequence = Config.train_max_depth
        max_depth_in_sequence = Config.train_min_depth
        intermediate_depths = []
        intermediate_images = []
        for i in range(len(raw_images)):
            depth = (raw_depths[i]).astype(np.float32) / 1000.0
            depth_nan = depth == np.nan
            depth_inf = depth == np.inf
            depth_nan_or_inf = np.logical_or(depth_inf, depth_nan)
            depth[depth_nan_or_inf] = 0
            depth = preprocessor.apply_depth(depth)
            intermediate_depths.append(depth)

            valid_mask = depth > 0
            valid_depth_values = depth[valid_mask]

            if len(valid_depth_values) > 0:
                current_min_depth = np.min(valid_depth_values)
                current_max_depth = np.max(valid_depth_values)
                min_depth_in_sequence = min(min_depth_in_sequence, current_min_depth)
                max_depth_in_sequence = max(max_depth_in_sequence, current_max_depth)

            image = raw_images[i]
            image = preprocessor.apply_rgb(image=image,
                                           scale_rgb=1.0,
                                           mean_rgb=[0.0, 0.0, 0.0],
                                           std_rgb=[1.0, 1.0, 1.0],
                                           normalize_colors=False)
            rgb_sum += np.sum(image)
            intermediate_images.append(image)
        rgb_average = rgb_sum / (len(raw_images) * Config.train_image_height * Config.train_image_width * 3)

        # GEOMETRIC AUGMENTATION
        geometric_scale_factor = 1.0
        if self.geometric_scale_augmentation:
            possible_low_scale_value = Config.train_min_depth / min_depth_in_sequence
            possible_high_scale_value = Config.train_max_depth / max_depth_in_sequence
            if np.random.random() > 0.5:
                low = max(possible_low_scale_value, 0.666)
                high = min(possible_high_scale_value, 1.5)
            else:
                low = max(possible_low_scale_value, 0.8)
                high = min(possible_high_scale_value, 1.25)
            geometric_scale_factor = np.random.uniform(low=low, high=high)

        # COLOR AUGMENTATION
        color_transforms = []
        brightness = random.uniform(-0.03, 0.03)
        contrast = random.uniform(0.8, 1.2)
        gamma = random.uniform(0.8, 1.2)
        color_transforms.append((adjust_gamma, gamma))
        color_transforms.append((adjust_contrast, contrast))
        color_transforms.append((adjust_brightness, brightness))
        random.shuffle(color_transforms)

        K = preprocessor.get_updated_intrinsics()
        for i in range(len(raw_images)):
            image = intermediate_images[i]
            depth = intermediate_depths[i] * geometric_scale_factor
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image.astype(np.float32))
            image = image / 255.0
            if self.split == "TRAINING" and (55.0 < rgb_average < 200.0):
                for (color_transform_function, color_transform_value) in color_transforms:
                    image = color_transform_function(image, color_transform_value)

            image = (image * 255.0) / self.scale_rgb
            image[0, :, :] = (image[0, :, :] - self.mean_rgb[0]) / self.std_rgb[0]
            image[1, :, :] = (image[1, :, :] - self.mean_rgb[1]) / self.std_rgb[1]
            image[2, :, :] = (image[2, :, :] - self.mean_rgb[2]) / self.std_rgb[2]

            pose = raw_poses[i].astype(np.float32)
            pose[0:3, 3] = pose[0:3, 3] * geometric_scale_factor
            pose = torch.from_numpy(pose)

            depth = torch.from_numpy(depth.astype(np.float32))

            output_poses.append(pose)
            output_depths.append(depth)
            output_images.append(image)

        K = torch.from_numpy(K.astype(np.float32))

        return output_images, output_depths, output_poses, K

    def __len__(self):
        return len(self.samples)


def main():
    subsequence_length = 8

    dataset = MVSDataset(
        root=Config.dataset,
        seed=Config.train_seed,
        split="TRAINING",
        subsequence_length=subsequence_length,
        scale_rgb=255.0,
        mean_rgb=[0.0, 0.0, 0.0],
        std_rgb=[1.0, 1.0, 1.0],
        geometric_scale_augmentation=False)

    print("Number of samples:", len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

    for i, (images, depths, poses, K) in enumerate(loader):
        for j in range(1, len(images)):
            current_image = images[j]
            current_depth = depths[j].unsqueeze(1)

            previous_image = images[j - 1]
            previous_depth = depths[j - 1].unsqueeze(1)

            print(np.max(current_depth.squeeze(1).numpy()[0]))
            print(np.min(current_depth.squeeze(1).numpy()[0]))
            current_image = (np.transpose(current_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
            current_depth = (current_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)
            measurement_image = (np.transpose(previous_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
            measurement_depth = (previous_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)

            cv2.imshow("Reference Image", cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
            cv2.imshow("Reference Depth", current_depth)
            cv2.imshow("Measurement Image", cv2.cvtColor(measurement_image, cv2.COLOR_BGR2RGB))
            cv2.imshow("Measurement Depth", measurement_depth)

            cv2.waitKey()


if __name__ == '__main__':
    main()
