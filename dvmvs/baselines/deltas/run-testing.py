import argparse

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from path import Path
from tqdm import tqdm

from dvmvs.baselines.deltas import superpoint, triangulation, densedepth
from dvmvs.baselines.deltas.utils import *
from dvmvs.config import Config
from dvmvs.dataset_loader import PreprocessImage, load_image
from dvmvs.utils import InferenceTimer, visualize_predictions, save_results

input_image_width = 320
input_image_height = 240
n_measurement_frames = 1

parser = argparse.ArgumentParser(description='DELTAS Inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: sequential: sequential folders')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', default=200, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=1, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float, default=10., help='maximum depth')
parser.add_argument('--width', type=int, default=input_image_width, help='image width')
parser.add_argument('--height', type=int, default=input_image_height, help='image height')
parser.add_argument('--seq_length', default=n_measurement_frames + 1, type=int, help='length of sequence')
parser.add_argument('--seq_gap', default=1, type=int, help='gap between frames for ScanNet dataset')
parser.add_argument('--resume', type=bool, default=True, help='Use pretrained network')
parser.add_argument('--pretrained', dest='pretrained', default='weights/pretrained_deltas', metavar='PATH', help='path to pre-trained model')
parser.add_argument('--do_confidence', type=bool, default=True, help='confidence in triangulation')
parser.add_argument('--dist_orthogonal', type=int, default=1, help='offset distance in pixels')
parser.add_argument('--kernel_size', type=int, default=1, help='kernel size')
parser.add_argument('--out_length', type=int, default=100, help='output length of epipolar patch')
parser.add_argument('--depth_range', type=bool, default=True, help='clamp using range of depth')
parser.add_argument('--num_kps', default=512, type=int, help='number of interest keypoints')
parser.add_argument('--model_type', type=str, default='resnet50', help='network backbone')
parser.add_argument('--align_corners', type=bool, default=False, help='align corners')
parser.add_argument('--descriptor_dim', type=int, default=128, help='dimension of descriptor')
parser.add_argument('--detection_threshold', type=float, default=0.0005, help='threshold for interest point detection')
parser.add_argument('--frac_superpoint', type=float, default=.5, help='fraction of interest points')
parser.add_argument('--nms_radius', type=int, default=9, help='radius for nms')

n_iter = 0


def get_model():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # create model
    print("=> creating model")

    # step 1 using superpoint
    config_sp = {
        'top_k_keypoints': args.num_kps,
        'height': args.height,
        'width': args.width,
        'align_corners': args.align_corners,
        'detection_threshold': args.detection_threshold,
        'frac_superpoint': args.frac_superpoint,
        'nms_radius': args.nms_radius,
    }

    cudnn.benchmark = True

    supernet = superpoint.Superpoint(config_sp)
    supernet = supernet.cuda() if torch.cuda.is_available() else supernet

    # step 2 using differentiable triangulation
    config_tri = {
        'dist_ortogonal': args.dist_orthogonal,
        'kernel_size': args.kernel_size,
        'out_length': args.out_length,
        'depth_range': args.depth_range,
        'has_confidence': args.do_confidence,
        'align_corners': args.align_corners,
    }

    trinet = triangulation.TriangulationNet(config_tri)
    trinet = trinet.cuda() if torch.cuda.is_available() else trinet

    # step 3 using sparse-to-dense

    config_depth = {
        'min_depth': args.mindepth,
        'max_depth': args.maxdepth,
        'input_shape': (args.height, args.width, 1),
    }

    depthnet = densedepth.SparsetoDenseNet(config_depth)
    depthnet = depthnet.cuda() if torch.cuda.is_available() else depthnet

    # load pre-trained weights

    if args.resume:
        if torch.cuda.is_available():
            weights = torch.load(args.pretrained)
        else:
            weights = torch.load(args.pretrained, map_location=torch.device('cpu'))
        supernet.load_state_dict(weights['state_dict'], strict=True)
        trinet.load_state_dict(weights['state_dict_tri'], strict=True)
        depthnet.load_state_dict(weights['state_dict_depth'], strict=True)
        if torch.cuda.is_available():
            depthnet = torch.nn.DataParallel(depthnet).cuda()
            supernet = torch.nn.DataParallel(supernet).cuda()
            trinet = torch.nn.DataParallel(trinet).cuda()

    return args, supernet, trinet, depthnet


def predict_for_subsequence(args, supernet, trinet, depthnet, tgt_img, tgt_depth, ref_imgs, ref_depths, poses, intrinsics):
    global n_iter

    tgt_img_var = tgt_img
    ref_imgs_var = ref_imgs
    img_var = make_symmetric(tgt_img_var, ref_imgs_var)
    batch_sz = tgt_img_var.shape[0]

    ##Pose and intrinsics
    poses_var = [pose for pose in poses]
    intrinsics_var = intrinsics
    seq_val = args.seq_length - 1
    pose = torch.cat(poses_var, 1)
    pose = pose_square(pose)

    ##Depth
    tgt_depth_var = tgt_depth
    ref_depths_var = [ref_depth for ref_depth in ref_depths]
    depth = tgt_depth_var
    depth_ref = torch.stack(ref_depths_var, 1)

    ##Step 1: Detect and Describe Points
    data_sp = {'img': img_var, 'process_tsp': 'ts'}  # t is detector, s is descriptor
    pred_sp = supernet(data_sp)

    # Keypoints and descriptor logic
    keypoints = pred_sp['keypoints'][:batch_sz]
    features = pred_sp['features'][:batch_sz]
    skip_half = pred_sp['skip_half'][:batch_sz]
    skip_quarter = pred_sp['skip_quarter'][:batch_sz]
    skip_eight = pred_sp['skip_eight'][:batch_sz]
    skip_sixteenth = pred_sp['skip_sixteenth'][:batch_sz]
    scores = pred_sp['scores'][:batch_sz]
    desc = pred_sp['descriptors']
    desc_anc = desc[:batch_sz, :, :, :]
    desc_view = desc[batch_sz:, :, :, :]
    desc_view = reorder_desc(desc_view, batch_sz)

    ## Step 2: Match & Triangulate Points
    data_sd = {'iter': n_iter, 'intrinsics': intrinsics_var, 'pose': pose, 'depth': depth, 'ref_depths': depth_ref, 'scores': scores,
               'keypoints': keypoints, 'descriptors': desc_anc, 'descriptors_views': desc_view, 'img_shape': tgt_img_var.shape,
               'sequence_length': seq_val}
    pred_sd = trinet(data_sd)

    view_matches = pred_sd['multiview_matches']
    anchor_keypoints = pred_sd['keypoints']
    keypoints3d_gt = pred_sd['keypoints3d_gt']
    range_mask_view = pred_sd['range_kp']
    range_mask = torch.sum(range_mask_view, 1)

    d_shp = tgt_depth_var.shape
    keypoints_3d = pred_sd['keypoints_3d']
    kp3d_val = keypoints_3d[:, :, 2].view(-1, 1).t()
    kp3d_filter = (range_mask > 0).view(-1, 1).t()
    kp3d_filter = (kp3d_filter) & (kp3d_val > args.mindepth) & (kp3d_val < args.maxdepth)

    ## Step 3: Densify using Sparse-to-Dense
    data_dd = {'anchor_keypoints': keypoints, 'keypoints_3d': keypoints_3d, 'sequence_length': args.seq_length, 'skip_sixteenth': skip_sixteenth,
               'range_mask': range_mask, 'features': features, 'skip_half': skip_half, 'skip_quarter': skip_quarter, 'skip_eight': skip_eight}
    pred_dd = depthnet(data_dd)
    output = pred_dd['dense_depth']

    return output


def predict():
    print("System: DELTAS")

    device = torch.device('cuda')
    cudnn.benchmark = True

    args, supernet, trinet, depthnet = get_model()

    supernet.eval()
    trinet.eval()
    depthnet.eval()

    scale_rgb = 255.0
    mean_rgb = [0.5, 0.5, 0.5]
    std_rgb = [0.5, 0.5, 0.5]

    dummy_input = torch.empty(size=(1, input_image_height, input_image_width), dtype=torch.float).to(device)

    data_path = Path(Config.test_offline_data_path)
    if Config.test_dataset_name is None:
        keyframe_index_files = sorted((Path(Config.test_offline_data_path) / "indices").files("*nmeas+{}*".format(n_measurement_frames)))
    else:
        keyframe_index_files = sorted((Path(Config.test_offline_data_path) / "indices").files("*" + Config.test_dataset_name + "*nmeas+{}*".format(n_measurement_frames)))
    for iteration, keyframe_index_file in enumerate(keyframe_index_files):
        keyframing_type, dataset_name, scene_name, _, _ = keyframe_index_file.split("/")[-1].split("+")

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

                # DELTAS ALWAYS REQUIRE A PREDETERMINED NUMBER OF MEASUREMENT FRAMES, SO FAKE IT
                while len(measurement_indices) < n_measurement_frames:
                    measurement_indices.append(measurement_indices[0])

                measurement_poses_torch = []
                measurement_images_torch = []
                for measurement_index in measurement_indices:
                    measurement_image = load_image(image_filenames[measurement_index])
                    measurement_image = preprocessor.apply_rgb(image=measurement_image,
                                                               scale_rgb=scale_rgb,
                                                               mean_rgb=mean_rgb,
                                                               std_rgb=std_rgb)
                    measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                    measurement_pose = poses[measurement_index]
                    measurement_pose = (np.linalg.inv(measurement_pose) @ reference_pose)
                    measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0).unsqueeze(0)
                    measurement_poses_torch.append(measurement_pose_torch)
                    measurement_images_torch.append(measurement_image_torch)

                K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

                tgt_depth = dummy_input
                ref_depths = [dummy_input for _ in range(n_measurement_frames)]

                inference_timer.record_start_time()
                prediction = predict_for_subsequence(args, supernet, trinet, depthnet,
                                                     tgt_img=reference_image_torch,
                                                     tgt_depth=tgt_depth,
                                                     ref_imgs=measurement_images_torch,
                                                     ref_depths=ref_depths,
                                                     poses=measurement_poses_torch,
                                                     intrinsics=K_torch)
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

        system_name = "{}_{}_{}_{}_{}_deltas".format(keyframing_type,
                                                     dataset_name,
                                                     input_image_width,
                                                     input_image_height,
                                                     n_measurement_frames)

        save_results(predictions=predictions,
                     groundtruths=reference_depths,
                     system_name=system_name,
                     scene_name=scene_name,
                     save_folder=Config.test_result_folder)


if __name__ == '__main__':
    predict()
