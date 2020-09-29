import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from dvmvs.baselines.mvdepthnet.decoder import Decoder
from dvmvs.baselines.mvdepthnet.encoder import Encoder
from dvmvs.config import Config
from dvmvs.dataset_loader import PreprocessImage, load_image
from dvmvs.utils import cost_volume_fusion, save_results, InferenceTimer, visualize_predictions, get_warp_grid_for_cost_volume_calculation


def predict():
    predict_with_finetuned = True

    if predict_with_finetuned:
        extension = "finetuned"
    else:
        extension = "without_ft"

    input_image_width = 320
    input_image_height = 256

    print("System: MVDEPTHNET, is_finetuned = ", predict_with_finetuned)

    device = torch.device('cuda')
    encoder = Encoder()
    decoder = Decoder()

    if predict_with_finetuned:
        encoder_weights = torch.load(Path("finetuned-weights").files("*encoder*")[0])
        decoder_weights = torch.load(Path("finetuned-weights").files("*decoder*")[0])
    else:
        mvdepth_weights = torch.load(Path("original-weights") / "pretrained_mvdepthnet_combined")
        pretrained_dict = mvdepth_weights['state_dict']
        encoder_weights = encoder.state_dict()
        pretrained_dict_encoder = {k: v for k, v in pretrained_dict.items() if k in encoder_weights}
        encoder_weights.update(pretrained_dict_encoder)
        decoder_weights = decoder.state_dict()
        pretrained_dict_decoder = {k: v for k, v in pretrained_dict.items() if k in decoder_weights}
        decoder_weights.update(pretrained_dict_decoder)

    encoder.load_state_dict(encoder_weights)
    decoder.load_state_dict(decoder_weights)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    warp_grid = get_warp_grid_for_cost_volume_calculation(width=input_image_width,
                                                          height=input_image_height,
                                                          device=device)

    min_depth = 0.5
    max_depth = 50.0
    n_depth_levels = 64

    scale_rgb = 1.0
    mean_rgb = [81.0, 81.0, 81.0]
    std_rgb = [35.0, 35.0, 35.0]

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
                                               new_width=input_image_width,
                                               new_height=input_image_height,
                                               distortion_crop=0,
                                               perform_crop=False)

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

                inference_timer.record_start_time()
                cost_volume = cost_volume_fusion(image1=reference_image_torch,
                                                 image2s=measurement_images_torch,
                                                 pose1=reference_pose_torch,
                                                 pose2s=measurement_poses_torch,
                                                 K=full_K_torch,
                                                 warp_grid=warp_grid,
                                                 min_depth=min_depth,
                                                 max_depth=max_depth,
                                                 n_depth_levels=n_depth_levels,
                                                 device=device,
                                                 dot_product=False)

                conv5, conv4, conv3, conv2, conv1 = encoder(reference_image_torch, cost_volume)
                prediction, _, _, _ = decoder(conv5, conv4, conv3, conv2, conv1)
                prediction = torch.clamp(prediction, min=0.02, max=2.0)
                prediction = 1 / prediction

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

        system_name = "{}_{}_{}_{}_{}_mvdepthnet_{}".format(keyframing_type,
                                                            dataset_name,
                                                            input_image_width,
                                                            input_image_height,
                                                            n_measurement_frames,
                                                            extension)

        save_results(predictions=predictions,
                     groundtruths=reference_depths,
                     system_name=system_name,
                     scene_name=scene_name,
                     save_folder=Config.test_result_folder)


if __name__ == '__main__':
    predict()
