from __future__ import division

import os
import zipfile

import cv2
import kornia
import numpy as np
import torch
from path import Path
from pytorch3d import structures, renderer

from dvmvs.errors import compute_errors


# GEOMETRIC UTILS
def pose_distance(reference_pose, measurement_pose):
    """
    :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose (not extrinsic matrix!)
    :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world pose (not extrinsic matrix!)
    :return combined_measure: float, combined pose distance measure
    :return R_measure: float, rotation distance measure
    :return t_measure: float, translation distance measure
    """
    rel_pose = np.dot(np.linalg.inv(reference_pose), measurement_pose)
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]
    R_measure = np.sqrt(2 * (1 - min(3.0, np.matrix.trace(R)) / 3))
    t_measure = np.linalg.norm(t)
    combined_measure = np.sqrt(t_measure ** 2 + R_measure ** 2)
    return combined_measure, R_measure, t_measure


def get_warp_grid_for_cost_volume_calculation(width, height, device):
    x = np.linspace(0, width - 1, num=int(width))
    y = np.linspace(0, height - 1, num=int(height))
    ones = np.ones(shape=(height, width))
    x_grid, y_grid = np.meshgrid(x, y)
    warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
    warp_grid = torch.from_numpy(warp_grid).float()
    warp_grid = warp_grid.view(-1, 3).t().to(device)
    return warp_grid


def calculate_cost_volume_by_warping(image1, image2, pose1, pose2, K, warp_grid, min_depth, max_depth, n_depth_levels, device, dot_product):
    batch_size, channels, height, width = image1.size()
    warp_grid = torch.cat(batch_size * [warp_grid.unsqueeze(dim=0)])

    cost_volume = torch.empty(size=(batch_size, n_depth_levels, height, width), dtype=torch.float32).to(device)

    extrinsic2 = torch.inverse(pose2).bmm(pose1)
    R = extrinsic2[:, 0:3, 0:3]
    t = extrinsic2[:, 0:3, 3].unsqueeze(-1)

    Kt = K.bmm(t)
    K_R_Kinv = K.bmm(R).bmm(torch.inverse(K))
    K_R_Kinv_UV = K_R_Kinv.bmm(warp_grid)

    inverse_depth_base = 1.0 / max_depth
    inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

    width_normalizer = width / 2.0
    height_normalizer = height / 2.0

    for depth_i in range(n_depth_levels):
        this_depth = 1 / (inverse_depth_base + depth_i * inverse_depth_step)

        warping = K_R_Kinv_UV + (Kt / this_depth)
        warping = warping.transpose(dim0=1, dim1=2)
        warping = warping[:, :, 0:2] / (warping[:, :, 2].unsqueeze(-1) + 1e-8)
        warping = warping.view(batch_size, height, width, 2)
        warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
        warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer

        warped_image2 = torch.nn.functional.grid_sample(input=image2,
                                                        grid=warping,
                                                        mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=True)

        if dot_product:
            cost_volume[:, depth_i, :, :] = torch.sum(image1 * warped_image2, dim=1) / channels
        else:
            cost_volume[:, depth_i, :, :] = torch.sum(torch.abs(image1 - warped_image2), dim=1)

    return cost_volume


def cost_volume_fusion(image1, image2s, pose1, pose2s, K, warp_grid, min_depth, max_depth, n_depth_levels, device, dot_product):
    batch_size, channels, height, width = image1.size()
    fused_cost_volume = torch.zeros(size=(batch_size, n_depth_levels, height, width), dtype=torch.float32).to(device)

    for pose2, image2 in zip(pose2s, image2s):
        cost_volume = calculate_cost_volume_by_warping(image1=image1,
                                                       image2=image2,
                                                       pose1=pose1,
                                                       pose2=pose2,
                                                       K=K,
                                                       warp_grid=warp_grid,
                                                       min_depth=min_depth,
                                                       max_depth=max_depth,
                                                       n_depth_levels=n_depth_levels,
                                                       device=device,
                                                       dot_product=dot_product)
        fused_cost_volume += cost_volume
    fused_cost_volume /= len(pose2s)
    return fused_cost_volume


def get_non_differentiable_rectangle_depth_estimation(reference_pose_torch,
                                                      measurement_pose_torch,
                                                      previous_depth_torch,
                                                      full_K_torch,
                                                      half_K_torch,
                                                      original_width,
                                                      original_height):
    batch_size, _, _ = reference_pose_torch.shape
    half_width = int(original_width / 2)
    half_height = int(original_height / 2)

    trans = torch.bmm(torch.inverse(reference_pose_torch), measurement_pose_torch)
    points_3d_src = kornia.depth_to_3d(previous_depth_torch, full_K_torch, normalize_points=False)
    points_3d_src = points_3d_src.permute(0, 2, 3, 1)
    points_3d_dst = kornia.transform_points(trans[:, None], points_3d_src)

    points_3d_dst = points_3d_dst.view(batch_size, -1, 3)

    z_values = points_3d_dst[:, :, -1]
    z_values = torch.relu(z_values)
    sorting_indices = torch.argsort(z_values, descending=True)
    z_values = torch.gather(z_values, dim=1, index=sorting_indices)

    sorting_indices_for_points = torch.stack([sorting_indices] * 3, dim=-1)
    points_3d_dst = torch.gather(points_3d_dst, dim=1, index=sorting_indices_for_points)

    projections = torch.round(kornia.project_points(points_3d_dst, half_K_torch.unsqueeze(1))).long()
    is_valid_below = (projections[:, :, 0] >= 0) & (projections[:, :, 1] >= 0)
    is_valid_above = (projections[:, :, 0] < half_width) & (projections[:, :, 1] < half_height)
    is_valid = is_valid_below & is_valid_above

    depth_hypothesis = torch.zeros(size=(batch_size, 1, half_height, half_width)).cuda()
    for projection_index in range(0, batch_size):
        valid_points_zs = z_values[projection_index][is_valid[projection_index]]
        valid_projections = projections[projection_index][is_valid[projection_index]]
        i_s = valid_projections[:, 1]
        j_s = valid_projections[:, 0]
        ij_combined = i_s * half_width + j_s
        _, ij_combined_unique_indices = np.unique(ij_combined.cpu().numpy(), return_index=True)
        ij_combined_unique_indices = torch.from_numpy(ij_combined_unique_indices).long().cuda()
        i_s = i_s[ij_combined_unique_indices]
        j_s = j_s[ij_combined_unique_indices]
        valid_points_zs = valid_points_zs[ij_combined_unique_indices]
        torch.index_put_(depth_hypothesis[projection_index, 0], (i_s, j_s), valid_points_zs)
    return depth_hypothesis


def get_differentiable_square_depth_estimation(reference_pose_torch,
                                               measurement_pose_torch,
                                               previous_depth_torch,
                                               full_K_torch,
                                               half_K_torch,
                                               original_image_size,
                                               device):
    batch_size, _, _ = full_K_torch.size()
    R_render = torch.eye(3, dtype=torch.float, device=device)
    T_render = torch.zeros(3, dtype=torch.float, device=device)
    R_render = torch.stack(batch_size * [R_render], dim=0)
    T_render = torch.stack(batch_size * [T_render], dim=0)
    R_render[:, 0, 0] *= -1
    R_render[:, 1, 1] *= -1

    trans = torch.bmm(torch.inverse(reference_pose_torch), measurement_pose_torch)
    points_3d_src = kornia.depth_to_3d(previous_depth_torch, full_K_torch, normalize_points=False)
    points_3d_src = points_3d_src.permute(0, 2, 3, 1)
    points_3d_dst = kornia.transform_points(trans[:, None], points_3d_src).view(batch_size, -1, 3)
    point_cloud_p3d = structures.Pointclouds(points=points_3d_dst, features=None)

    width_normalizer = original_image_size / 4.0
    height_normalizer = original_image_size / 4.0
    px_ndc = (half_K_torch[:, 0, 2] - width_normalizer) / width_normalizer
    py_ndc = (half_K_torch[:, 1, 2] - height_normalizer) / height_normalizer
    fx_ndc = half_K_torch[:, 0, 0] / width_normalizer
    fy_ndc = half_K_torch[:, 1, 1] / height_normalizer

    principal_point = torch.stack([px_ndc, py_ndc], dim=-1)
    focal_length = torch.stack([fx_ndc, fy_ndc], dim=-1)

    cameras = renderer.SfMPerspectiveCameras(focal_length=focal_length,
                                             principal_point=principal_point,
                                             R=R_render,
                                             T=T_render,
                                             device=torch.device('cuda'))

    raster_settings = renderer.PointsRasterizationSettings(
        image_size=int(original_image_size / 2.0),
        radius=0.02,
        points_per_pixel=3)

    depth_renderer = renderer.PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    rendered_depth = torch.min(depth_renderer(point_cloud_p3d).zbuf, dim=-1)[0]
    depth_hypothesis = torch.relu(rendered_depth).unsqueeze(1)
    return depth_hypothesis


def warp_frame_depth(
        image_src: torch.Tensor,
        depth_dst: torch.Tensor,
        src_trans_dst: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize_points: bool = False,
        sampling_mode='bilinear') -> torch.Tensor:
    # TAKEN FROM KORNIA LIBRARY
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. "
                        f"Got {type(src_trans_dst)}.")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                         f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = kornia.depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = kornia.transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3
    points_3d_src[:, :, :, 2] = torch.relu(points_3d_src[:, :, :, 2])

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = kornia.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = kornia.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    return torch.nn.functional.grid_sample(image_src, points_2d_src_norm, align_corners=True, mode=sampling_mode)


def is_pose_available(pose):
    is_nan = np.isnan(pose).any()
    is_inf = np.isinf(pose).any()
    is_neg_inf = np.isneginf(pose).any()
    if is_nan or is_inf or is_neg_inf:
        return False
    else:
        return True


# TRAINING UTILS
def freeze_batchnorm(module):
    if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
        module.eval()
        module.weight.requires_grad = False
        module.bias.requires_grad = False


def zip_code(run_directory):
    zip_file_path = os.path.join(run_directory, "code.zip")
    zip_handle = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)

    files = Path("./").files("*.py")
    for file in files:
        zip_handle.write(file)

    files = Path("../").files("*.py")
    for file in files:
        zip_handle.write(file)

    zip_handle.close()


def save_checkpoint(save_path, models, step, loss, filename='checkpoint.pth.tar'):
    save_path = Path(save_path)
    for model in models:
        prefix = model['name']
        model_state = model['state_dict']
        torch.save(model_state, save_path / '{}_{}_epoch:{}_l1:{:.4f}_l1-inv:{:.4f}_l1-rel:{:.4f}_huber:{:.4f}'.format(prefix,
                                                                                                                       filename,
                                                                                                                       step,
                                                                                                                       loss[0],
                                                                                                                       loss[1],
                                                                                                                       loss[2],
                                                                                                                       loss[3]))


def save_optimizer(save_path, optimizer, step, loss, filename='checkpoint.pth.tar'):
    save_path = Path(save_path)
    optimizer_state = optimizer.state_dict()
    torch.save(optimizer_state, save_path / 'optimizer_{}_epoch:{}_l1:{:.4f}_l1-inv:{:.4f}_l1-rel:{:.4f}_huber:{:.4f}'.format(filename,
                                                                                                                              step,
                                                                                                                              loss[0],
                                                                                                                              loss[1],
                                                                                                                              loss[2],
                                                                                                                              loss[3]))


def print_number_of_trainable_parameters(optimizer):
    parameter_counter = 0
    for param_group in optimizer.param_groups:
        for parameter in param_group['params']:
            if parameter.requires_grad:
                parameter_counter += parameter.nelement()

    print("Number of trainable parameters:", f"{parameter_counter:,d}")


# TESTING UTILS
def save_results(predictions, groundtruths, system_name, scene_name, save_folder, max_depth=np.inf):
    if groundtruths is not None:
        errors = []
        for i, prediction in enumerate(predictions):
            errors.append(compute_errors(groundtruths[i], prediction, max_depth))

        error_names = ['abs_error', 'abs_relative_error', 'abs_inverse_error',
                       'squared_relative_error', 'rmse', 'ratio_125', 'ratio_125_2', 'ratio_125_3']

        errors = np.array(errors)
        mean_errors = np.nanmean(errors, 0)
        print("Metrics of {} for scene {}:".format(system_name, scene_name))
        print("{:>25}, {:>25}, {:>25}, {:>25}, {:>25}, {:>25}, {:>25}, {:>25}".format(*error_names))
        print("{:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}, {:25.4f}".format(*mean_errors))

        np.savez_compressed(Path(save_folder) / system_name + "_errors_" + scene_name, errors)

    predictions = np.array(predictions)
    np.savez_compressed(Path(save_folder) / system_name + "_predictions_" + scene_name, predictions)


def save_predictions(predictions, system_name, scene_name, save_folder):
    np.savez_compressed(Path(save_folder) / system_name + "_predictions_" + scene_name, predictions)


def visualize_predictions(numpy_reference_image, numpy_measurement_image, numpy_predicted_depth, normalization_mean, normalization_std, normalization_scale,
                          depth_multiplier_for_visualization=5000):
    numpy_reference_image = numpy_reference_image * np.array(normalization_std) + np.array(normalization_mean)
    numpy_reference_image = (numpy_reference_image * normalization_scale).astype(np.uint8)

    numpy_measurement_image = numpy_measurement_image * np.array(normalization_std) + np.array(normalization_mean)
    numpy_measurement_image = (numpy_measurement_image * normalization_scale).astype(np.uint8)

    cv2.imshow("Reference Image", cv2.cvtColor(numpy_reference_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("A Measurement Image", cv2.cvtColor(numpy_measurement_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Predicted Depth", (depth_multiplier_for_visualization * numpy_predicted_depth).astype(np.uint16))
    cv2.waitKey()


class InferenceTimer:
    def __init__(self, n_skip=20):
        self.times = []
        self.n_skip = n_skip

        self.forward_pass_start = torch.cuda.Event(enable_timing=True)
        self.forward_pass_end = torch.cuda.Event(enable_timing=True)

    def record_start_time(self):
        self.forward_pass_start.record()

    def record_end_time_and_elapsed_time(self):
        self.forward_pass_end.record()
        torch.cuda.synchronize()

        elapsed_time = self.forward_pass_start.elapsed_time(self.forward_pass_end)
        self.times.append(elapsed_time)

    def print_statistics(self):
        times = np.array(self.times[self.n_skip:])
        if len(times) > 0:
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            median_time = np.median(times)
            print("Number of Forward Passes:", len(times))
            print("--- Mean Inference Time:", mean_time)
            print("--- Std Inference Time:", std_time)
            print("--- Median Inference Time:", median_time)
            print("--- Min Inference Time:", min_time)
            print("--- Max Inference Time:", max_time)
        else:
            print("Not enough time measurements are taken!")
