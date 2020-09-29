from __future__ import division

import torch
from torch import nn


class LossMeter(object):
    def __init__(self):
        self.count = 0.0
        self.sum = 0.0
        self.avg = 0.0

        self.item_average = 0.0

    def update(self, loss, count):
        self.sum += loss
        self.count += count
        self.avg = self.sum / self.count

        self.item_average = loss / count

    def __repr__(self):
        return '{:.4f} ({:.4f})'.format(self.item_average, self.avg)


def update_losses(predictions, weights, groundtruth, is_training, l1_meter, huber_meter, l1_inv_meter, l1_rel_meter, loss_type):
    optimizer_loss = 0
    sample_l1_loss, sample_huber_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = None, None, None, None, None
    if is_training:
        for j, prediction in enumerate(predictions):
            sample_l1_loss, sample_huber_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = \
                calculate_loss(groundtruth=groundtruth, prediction=prediction)

            if loss_type == "L1":
                optimizer_loss = optimizer_loss + weights[j] * (sample_l1_loss / sample_valid_count)
            elif loss_type == "L1-inv":
                optimizer_loss = optimizer_loss + weights[j] * (sample_l1_inv_loss / sample_valid_count)
            elif loss_type == "L1-rel":
                optimizer_loss = optimizer_loss + weights[j] * (sample_l1_rel_loss / sample_valid_count)
            elif loss_type == "Huber":
                optimizer_loss = optimizer_loss + weights[j] * (sample_huber_loss / sample_valid_count)
    else:
        sample_l1_loss, sample_huber_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = calculate_loss(groundtruth=groundtruth,
                                                                                                                       prediction=predictions[-1])
    l1_meter.update(sample_l1_loss.item(), sample_valid_count)
    huber_meter.update(sample_huber_loss.item(), sample_valid_count)
    l1_inv_meter.update(sample_l1_inv_loss.item(), sample_valid_count)
    l1_rel_meter.update(sample_l1_rel_loss.item(), sample_valid_count)

    return optimizer_loss


def calculate_loss(groundtruth, prediction):
    batch, height_original, width_original = groundtruth.size()
    groundtruth = groundtruth.view(batch, 1, height_original, width_original)

    batch, height_scaled, width_scaled = prediction.size()
    prediction = prediction.view(batch, 1, height_scaled, width_scaled)

    groundtruth_scaled = nn.functional.interpolate(groundtruth,
                                                   size=(height_scaled, width_scaled),
                                                   mode='nearest')

    valid_mask = groundtruth_scaled != 0
    valid_count = valid_mask.nonzero().size()[0]

    groundtruth_valid = groundtruth_scaled[valid_mask]
    prediction_valid = prediction[valid_mask]

    groundtruth_inverse_valid = 1.0 / groundtruth_valid
    prediction_inverse_valid = 1.0 / prediction_valid

    l1_diff = torch.abs(groundtruth_valid - prediction_valid)

    smooth_l1_loss = torch.nn.functional.smooth_l1_loss(prediction_valid, groundtruth_valid, reduction='none')
    smooth_l1_loss = torch.sum(smooth_l1_loss)

    l1_loss = torch.sum(l1_diff)
    l1_inv_loss = torch.sum(torch.abs(groundtruth_inverse_valid - prediction_inverse_valid))
    l1_rel_loss = torch.sum(l1_diff / groundtruth_valid)

    return l1_loss, smooth_l1_loss, l1_inv_loss, l1_rel_loss, valid_count
