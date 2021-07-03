import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from dvmvs.config import Config
from dvmvs.dataset_loader import PreprocessImage, load_image
from dvmvs.keyframe_buffer import KeyframeBuffer
from dvmvs.pairnet.model import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, CostVolumeDecoder
from dvmvs.utils import cost_volume_fusion, save_results, visualize_predictions, InferenceTimer, get_warp_grid_for_cost_volume_calculation


def predict():
    dataset_name = Config.test_online_scene_path.split("/")[-2]
    system_name = "keyframe_{}_{}_{}_{}_dvmvs_fusionnet_online".format(dataset_name,
                                                                       Config.test_image_width,
                                                                       Config.test_image_height,
                                                                       Config.test_n_measurement_frames)

    print("Predicting with System:", system_name)
    print("# of Measurement Frames:", Config.test_n_measurement_frames)

    device = torch.device("cuda")
    feature_extractor = FeatureExtractor()
    feature_shrinker = FeatureShrinker()
    cost_volume_encoder = CostVolumeEncoder()
    cost_volume_decoder = CostVolumeDecoder()

    feature_extractor = feature_extractor.to(device)
    feature_shrinker = feature_shrinker.to(device)
    cost_volume_encoder = cost_volume_encoder.to(device)
    cost_volume_decoder = cost_volume_decoder.to(device)

    model = [feature_extractor, feature_shrinker, cost_volume_encoder, cost_volume_decoder]

    for i in range(len(model)):
        try:
            checkpoint = sorted(Path("weights").files())[i]
            weights = torch.load(checkpoint)
            model[i].load_state_dict(weights)
            model[i].eval()
            print("Loaded weights for", checkpoint)
        except Exception as e:
            print(e)
            print("Could not find the checkpoint for module", i)
            exit(1)

    feature_extractor = model[0]
    feature_shrinker = model[1]
    cost_volume_encoder = model[2]
    cost_volume_decoder = model[3]

    warp_grid = get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 2),
                                                          height=int(Config.test_image_height / 2),
                                                          device=device)

    scale_rgb = 255.0
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    min_depth = 0.25
    max_depth = 20.0
    n_depth_levels = 64

    scene_folder = Path(Config.test_online_scene_path)

    scene = scene_folder.split("/")[-1]
    print("Predicting for scene:", scene)

    keyframe_buffer = KeyframeBuffer(buffer_size=Config.test_keyframe_buffer_size,
                                     keyframe_pose_distance=Config.test_keyframe_pose_distance,
                                     optimal_t_score=Config.test_optimal_t_measure,
                                     optimal_R_score=Config.test_optimal_R_measure,
                                     store_return_indices=False)

    K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames = sorted((scene_folder / 'images').files("*.png"))
    depth_filenames = sorted((scene_folder / 'depth').files("*.png"))

    inference_timer = InferenceTimer()

    predictions = []
    reference_depths = []
    with torch.no_grad():
        for i in tqdm(range(0, len(poses))):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])
            reference_depth = cv2.imread(depth_filenames[i], -1).astype(float) / 1000.0

            # POLL THE KEYFRAME BUFFER
            response = keyframe_buffer.try_new_keyframe(reference_pose, reference_image)
            if response != 1:
                continue

            preprocessor = PreprocessImage(K=K,
                                           old_width=reference_image.shape[1],
                                           old_height=reference_image.shape[0],
                                           new_width=Config.test_image_width,
                                           new_height=Config.test_image_height,
                                           distortion_crop=Config.test_distortion_crop,
                                           perform_crop=Config.test_perform_crop)

            reference_image = preprocessor.apply_rgb(image=reference_image,
                                                     scale_rgb=scale_rgb,
                                                     mean_rgb=mean_rgb,
                                                     std_rgb=std_rgb)
            reference_depth = preprocessor.apply_depth(reference_depth)
            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

            full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

            half_K_torch = full_K_torch.clone().cuda()
            half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

            measurement_poses_torch = []
            measurement_images_torch = []
            measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
            for (measurement_pose, measurement_image) in measurement_frames:
                measurement_image = preprocessor.apply_rgb(image=measurement_image,
                                                           scale_rgb=scale_rgb,
                                                           mean_rgb=mean_rgb,
                                                           std_rgb=std_rgb)
                measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
                measurement_images_torch.append(measurement_image_torch)
                measurement_poses_torch.append(measurement_pose_torch)

            inference_timer.record_start_time()

            measurement_feature_halfs = []
            for measurement_image_torch in measurement_images_torch:
                measurement_feature_half, _, _, _ = feature_shrinker(*feature_extractor(measurement_image_torch))
                measurement_feature_halfs.append(measurement_feature_half)

            reference_feature_half, reference_feature_quarter, \
            reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))

            cost_volume = cost_volume_fusion(image1=reference_feature_half,
                                             image2s=measurement_feature_halfs,
                                             pose1=reference_pose_torch,
                                             pose2s=measurement_poses_torch,
                                             K=half_K_torch,
                                             warp_grid=warp_grid,
                                             min_depth=min_depth,
                                             max_depth=max_depth,
                                             n_depth_levels=n_depth_levels,
                                             device=device,
                                             dot_product=True)

            skip0, skip1, skip2, skip3, bottom = cost_volume_encoder(features_half=reference_feature_half,
                                                                     features_quarter=reference_feature_quarter,
                                                                     features_one_eight=reference_feature_one_eight,
                                                                     features_one_sixteen=reference_feature_one_sixteen,
                                                                     cost_volume=cost_volume)

            prediction, _, _, _, _ = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, bottom)

            inference_timer.record_end_time_and_elapsed_time()

            prediction = prediction.cpu().numpy().squeeze()
            reference_depths.append(reference_depth)
            predictions.append(prediction)

            if Config.test_visualize:
                visualize_predictions(numpy_reference_image=reference_image,
                                      numpy_measurement_image=measurement_image,
                                      numpy_predicted_depth=prediction,
                                      normalization_mean=mean_rgb,
                                      normalization_std=std_rgb,
                                      normalization_scale=scale_rgb)

        inference_timer.print_statistics()

        save_results(predictions=predictions,
                     groundtruths=reference_depths,
                     system_name=system_name,
                     scene_name=scene,
                     save_folder=".")


if __name__ == '__main__':
    predict()
