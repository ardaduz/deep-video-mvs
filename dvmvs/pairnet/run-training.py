import datetime
import itertools
import os

import numpy as np
from path import Path
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dvmvs.dataset_loader import MVSDataset
from dvmvs.losses import LossMeter, update_losses
from dvmvs.pairnet.model import *
from dvmvs.train import train
from dvmvs.utils import zip_code, print_number_of_trainable_parameters, calculate_cost_volume_by_warping


class TrainingHyperparameters:
    Config.train_subsequence_length = 2
    Config.train_predict_two_way = True
    batch_size = 14
    learning_rate = 1e-4
    momentum = 0.9
    beta = 0.999
    weight_decay = 0
    # loss_type = "Huber"
    # loss_type = "L1"
    loss_type = "L1-inv"
    # loss_type = "L1-rel"

    finetune_epochs = 2

    use_augmentation = True
    use_checkpoint = False


scaling = 0.5
x = np.linspace(0, Config.train_image_width * scaling - 1, num=int(Config.train_image_width * scaling))
y = np.linspace(0, Config.train_image_height * scaling - 1, num=int(Config.train_image_height * scaling))
ones = np.ones(shape=(int(Config.train_image_height * scaling), int(Config.train_image_width * scaling)))
x_grid, y_grid = np.meshgrid(x, y)
warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
warp_grid = torch.from_numpy(warp_grid).float()
warp_grid = warp_grid.view(-1, 3).t().cuda()


def main():
    # set the manual seed for reproducibility
    torch.manual_seed(Config.train_seed)

    # create the directory for this run of the training
    run_directory = os.path.join(Config.train_run_directory, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(run_directory)

    # zip every code file
    zip_code(run_directory)

    summary_writer = SummaryWriter(run_directory)

    print("=> fetching scenes in '{}'".format(Config.dataset))
    train_set = MVSDataset(
        root=Config.dataset,
        seed=Config.train_seed,
        split="TRAINING",
        subsequence_length=Config.train_subsequence_length,
        scale_rgb=255.0,
        mean_rgb=[0.485, 0.456, 0.406],
        std_rgb=[0.229, 0.224, 0.225],
        geometric_scale_augmentation=True
    )

    val_set = MVSDataset(
        root=Config.dataset,
        seed=Config.train_seed,
        split="VALIDATION",
        subsequence_length=Config.train_subsequence_length,
        scale_rgb=255.0,
        mean_rgb=[0.485, 0.456, 0.406],
        std_rgb=[0.229, 0.224, 0.225]
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = DataLoader(dataset=train_set,
                              batch_size=TrainingHyperparameters.batch_size,
                              shuffle=True,
                              num_workers=Config.train_data_pipeline_workers,
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=TrainingHyperparameters.batch_size,
                            shuffle=False,
                            num_workers=Config.train_data_pipeline_workers,
                            pin_memory=True,
                            drop_last=True)

    feature_extractor = FeatureExtractor()
    feature_shrinker = FeatureShrinker()
    cost_volume_encoder = CostVolumeEncoder()
    cost_volume_decoder = CostVolumeDecoder()
    feature_extractor = feature_extractor.cuda()
    feature_shrinker = feature_shrinker.cuda()
    cost_volume_encoder = cost_volume_encoder.cuda()
    cost_volume_decoder = cost_volume_decoder.cuda()
    model = [feature_extractor, feature_shrinker, cost_volume_encoder, cost_volume_decoder]

    if TrainingHyperparameters.use_checkpoint:
        for i in range(len(model)):
            try:
                checkpoint = Path(".").files("module_" + str(i) + "*")[0]
                weights = torch.load(checkpoint)
                model[i].load_state_dict(weights)
                print("Loaded weights for", checkpoint)
            except Exception as e:
                print(e)
                print("Skipping...")

    cudnn.benchmark = True

    best_loss = [np.inf, np.inf, np.inf, np.inf]

    # TRAIN MY PARTS
    parameters = itertools.chain(feature_shrinker.parameters(),
                                 cost_volume_encoder.parameters(),
                                 cost_volume_decoder.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 lr=TrainingHyperparameters.learning_rate,
                                 betas=(TrainingHyperparameters.momentum, TrainingHyperparameters.beta),
                                 weight_decay=TrainingHyperparameters.weight_decay)
    print_number_of_trainable_parameters(optimizer)
    for epoch in range(TrainingHyperparameters.finetune_epochs):
        print("\n\nEPOCH:", epoch)
        train(train_loader=train_loader,
              val_loader=val_loader,
              model=model,
              optimizer=optimizer,
              summary_writer=summary_writer,
              epoch=epoch,
              best_loss=best_loss,
              run_directory=run_directory,
              forward_pass_function=forward_pass)

    # TRAIN EVERYTHING
    parameters = itertools.chain(feature_extractor.parameters(),
                                 feature_shrinker.parameters(),
                                 cost_volume_encoder.parameters(),
                                 cost_volume_decoder.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 lr=TrainingHyperparameters.learning_rate,
                                 betas=(TrainingHyperparameters.momentum, TrainingHyperparameters.beta),
                                 weight_decay=TrainingHyperparameters.weight_decay)
    print_number_of_trainable_parameters(optimizer)
    for epoch in range(TrainingHyperparameters.finetune_epochs, Config.train_epochs):
        print("\n\nEPOCH:", epoch)
        train(train_loader=train_loader,
              val_loader=val_loader,
              model=model,
              optimizer=optimizer,
              summary_writer=summary_writer,
              epoch=epoch,
              best_loss=best_loss,
              run_directory=run_directory,
              forward_pass_function=forward_pass)


def forward_pass(images, depths, poses, K, model, is_training):
    feature_extractor = model[0]
    feature_shrinker = model[1]
    cost_volume_encoder = model[2]
    cost_volume_decoder = model[3]

    K[:, 0:2, :] = K[:, 0:2, :] * scaling
    K = K.cuda()

    images_cuda = []
    depths_cuda = []
    poses_cuda = []
    feature_halfs = []
    feature_quarters = []
    feature_one_eights = []
    feature_one_sixteens = []
    # Extract image features
    for i in range(0, len(images)):
        images_cuda.append(images[i].cuda())
        poses_cuda.append(poses[i].cuda())
        depths_cuda.append(depths[i].cuda())
        feature_half, feature_quarter, feature_one_eight, feature_one_sixteen = feature_shrinker(*feature_extractor(images_cuda[i]))
        feature_halfs.append(feature_half)
        feature_quarters.append(feature_quarter)
        feature_one_eights.append(feature_one_eight)
        feature_one_sixteens.append(feature_one_sixteen)

    optimizer_loss = 0
    predictions = None
    l1_meter = LossMeter()
    huber_meter = LossMeter()
    l1_inv_meter = LossMeter()
    l1_rel_meter = LossMeter()
    for i in range(1, len(images)):
        reference_index = i
        measurement_index = i - 1

        if Config.train_predict_two_way:
            iterations = [[measurement_index, reference_index],
                          [reference_index, measurement_index]]
        else:
            iterations = [[reference_index, measurement_index]]

        for [index1, index2] in iterations:
            initial_cost_volume = calculate_cost_volume_by_warping(image1=feature_halfs[index1],
                                                                   image2=feature_halfs[index2],
                                                                   pose1=poses_cuda[index1],
                                                                   pose2=poses_cuda[index2],
                                                                   K=K,
                                                                   warp_grid=warp_grid,
                                                                   min_depth=Config.train_min_depth,
                                                                   max_depth=Config.train_max_depth,
                                                                   n_depth_levels=Config.train_n_depth_levels,
                                                                   device=torch.device('cuda'),
                                                                   dot_product=True)
            flipped = False
            to_be_used_feature_one_sixteen = feature_one_sixteens[index1]
            to_be_used_feature_one_eight = feature_one_eights[index1]
            to_be_used_feature_quarter = feature_quarters[index1]
            to_be_used_feature_half = feature_halfs[index1]
            to_be_used_image = images_cuda[index1]
            to_be_used_depth = depths_cuda[index1]
            to_be_used_cost_volume = initial_cost_volume
            if is_training and TrainingHyperparameters.use_augmentation and np.random.random() > 0.5:
                to_be_used_feature_one_sixteen = torch.flip(feature_one_sixteens[index1], dims=[-1])
                to_be_used_feature_one_eight = torch.flip(feature_one_eights[index1], dims=[-1])
                to_be_used_feature_quarter = torch.flip(feature_quarters[index1], dims=[-1])
                to_be_used_feature_half = torch.flip(feature_halfs[index1], dims=[-1])
                to_be_used_image = torch.flip(images_cuda[index1], dims=[-1])
                to_be_used_depth = torch.flip(depths_cuda[index1], dims=[-1])
                to_be_used_cost_volume = torch.flip(initial_cost_volume, dims=[-1])
                flipped = True

            skip0, skip1, skip2, skip3, bottom = cost_volume_encoder(features_half=to_be_used_feature_half,
                                                                     features_quarter=to_be_used_feature_quarter,
                                                                     features_one_eight=to_be_used_feature_one_eight,
                                                                     features_one_sixteen=to_be_used_feature_one_sixteen,
                                                                     cost_volume=to_be_used_cost_volume)

            depth_full, depth_half, depth_quarter, depth_one_eight, depth_one_sixteen = cost_volume_decoder(to_be_used_image,
                                                                                                            skip0,
                                                                                                            skip1,
                                                                                                            skip2,
                                                                                                            skip3,
                                                                                                            bottom)

            predictions = [depth_one_sixteen, depth_one_eight, depth_quarter, depth_half, depth_full]

            weights = [1, 1, 1, 1, 1]
            optimizer_loss = optimizer_loss + update_losses(predictions=predictions,
                                                            weights=weights,
                                                            groundtruth=to_be_used_depth,
                                                            is_training=is_training,
                                                            l1_meter=l1_meter,
                                                            huber_meter=huber_meter,
                                                            l1_inv_meter=l1_inv_meter,
                                                            l1_rel_meter=l1_rel_meter,
                                                            loss_type=TrainingHyperparameters.loss_type)

            if flipped and index1 == len(images) - 1:
                depth_quarter = torch.flip(depth_quarter, dims=[-1])
                depth_half = torch.flip(depth_half, dims=[-1])
                depth_full = torch.flip(depth_full, dims=[-1])

            predictions = [depth_quarter, depth_half, depth_full]

    predictions_names = ["prediction_quarter", "prediction_half", "prediction_full"]

    return l1_meter, huber_meter, l1_inv_meter, l1_rel_meter, optimizer_loss, predictions, predictions_names


if __name__ == '__main__':
    main()
