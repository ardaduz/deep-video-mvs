import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from dvmvs.config import Config
from dvmvs.dataset_loader import PreprocessImage, load_image
from dvmvs.pairnet.model import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, CostVolumeDecoder
from dvmvs.utils import cost_volume_fusion, save_results, visualize_predictions, InferenceTimer, get_warp_grid_for_cost_volume_calculation


def predict():
    print("System: PAIRNET")

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

    data_path = Path(Config.test_offline_data_path)
    if Config.test_dataset_name is None:
        keyframe_index_files = sorted((Path(Config.test_offline_data_path) / "indices").files())
    else:
        keyframe_index_files = sorted((Path(Config.test_offline_data_path) / "indices").files("*" + Config.test_dataset_name + "*"))
    for iteration, keyframe_index_file in enumerate(keyframe_index_files):
        keyframing_type, dataset_name, scene_name, _, n_measurement_frames = keyframe_index_file.split("/")[-1].split("+")

        scene_folder = data_path / dataset_name / scene_name
        print("Predicting for scene:", dataset_name + "-" + scene_name, " - ", iteration, "/", len(keyframe_index_files))

        keyframe_index_file_lines = np.loadtxt(keyframe_index_file, dtype=str, delimiter="\n")

        K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
        poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
        image_filenames = sorted((scene_folder / 'images').files("*.png"))
        depth_filenames = sorted((scene_folder / 'depth').files("*.png"))

        input_filenames = []
        for image_filename in image_filenames:
            input_filenames.append(image_filename.split("/")[-1])

        inference_timer = InferenceTimer()

        predictions = []
        reference_depths = []
        with torch.no_grad():
            for i in tqdm(range(0, len(keyframe_index_file_lines))):

                keyframe_index_file_line = keyframe_index_file_lines[i]

                if keyframe_index_file_line == "TRACKING LOST":
                    continue
                else:
                    current_input_filenames = keyframe_index_file_line.split(" ")
                    current_indices = [input_filenames.index(current_input_filenames[x]) for x in range(len(current_input_filenames))]

                reference_index = current_indices[0]
                measurement_indices = current_indices[1:]

                reference_pose = poses[reference_index]
                reference_image = load_image(image_filenames[reference_index])
                reference_depth = cv2.imread(depth_filenames[reference_index], -1).astype(float) / 1000.0

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

                measurement_poses_torch = []
                measurement_images_torch = []
                for measurement_index in measurement_indices:
                    measurement_image = load_image(image_filenames[measurement_index])
                    measurement_image = preprocessor.apply_rgb(image=measurement_image,
                                                               scale_rgb=scale_rgb,
                                                               mean_rgb=mean_rgb,
                                                               std_rgb=std_rgb)
                    measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                    measurement_pose_torch = torch.from_numpy(poses[measurement_index]).float().to(device).unsqueeze(0)
                    measurement_images_torch.append(measurement_image_torch)
                    measurement_poses_torch.append(measurement_pose_torch)

                full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

                half_K_torch = full_K_torch.clone().cuda()
                half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

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

        system_name = "{}_{}_{}_{}_{}_dvmvs_pairnet".format(keyframing_type,
                                                            dataset_name,
                                                            Config.test_image_width,
                                                            Config.test_image_height,
                                                            n_measurement_frames)

        save_results(predictions=predictions,
                     groundtruths=reference_depths,
                     system_name=system_name,
                     scene_name=scene_name,
                     save_folder=Config.test_result_folder)


if __name__ == '__main__':
    predict()
