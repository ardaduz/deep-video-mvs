import torch
import torchvision
from tqdm import tqdm

from dvmvs.config import Config
from dvmvs.losses import LossMeter
from dvmvs.utils import save_checkpoint, save_optimizer, freeze_batchnorm


def switch_mode(model, mode):
    if mode == 'train':
        for module in model:
            module.train()
            if Config.train_freeze_batch_normalization:
                module.apply(freeze_batchnorm)
    elif mode == 'eval':
        for module in model:
            module.eval()


def train(train_loader, val_loader, model, optimizer, summary_writer, epoch, best_loss, run_directory, forward_pass_function):
    training_l1_meter = LossMeter()
    training_huber_meter = LossMeter()
    training_l1_inv_meter = LossMeter()
    training_l1_rel_meter = LossMeter()

    info_printer = tqdm(total=0, position=1, bar_format='{desc}')
    info = 'L1 Loss: {} --- L1-inv Loss: {} --- L1-rel Loss: {} --- Huber Loss: {}'

    # switch to train mode
    switch_mode(model=model, mode='train')

    for i, (images, depths, poses, K) in enumerate(tqdm(train_loader)):
        batch_l1_meter, batch_huber_meter, batch_l1_inv_meter, batch_l1_rel_meter, \
        optimizer_loss, predictions, predictions_names = forward_pass_function(images=images,
                                                                               depths=depths,
                                                                               poses=poses,
                                                                               K=K,
                                                                               model=model,
                                                                               is_training=True)
        # record losses
        training_l1_meter.update(loss=batch_l1_meter.sum, count=batch_l1_meter.count)
        training_huber_meter.update(loss=batch_huber_meter.sum, count=batch_huber_meter.count)
        training_l1_inv_meter.update(loss=batch_l1_inv_meter.sum, count=batch_l1_inv_meter.count)
        training_l1_rel_meter.update(loss=batch_l1_rel_meter.sum, count=batch_l1_rel_meter.count)

        if i > 0 and i % Config.train_print_frequency == 0:
            rgb_debug_image = images[-1][0].cpu().detach()
            depth_debug_image = depths[-1][0].cpu().repeat(3, 1, 1).detach()
            debug_images = [rgb_debug_image, depth_debug_image]
            debug_names = "input_image   ground_truth"
            for index, prediction in enumerate(predictions):
                debug_names += "   " + predictions_names[index]
                prediction = prediction[0].cpu().repeat(3, 1, 1).detach().unsqueeze(0)
                _, channel, height, width = prediction.size()
                scale_factor = Config.train_image_width / width
                prediction = torch.nn.functional.interpolate(prediction, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                prediction = prediction.squeeze(0)
                debug_images.append(prediction)

            debug_images_grid = torchvision.utils.make_grid(debug_images,
                                                            nrow=3,
                                                            normalize=True,
                                                            scale_each=True)

            summary_writer.add_image(debug_names, debug_images_grid, epoch * len(train_loader) + i)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        optimizer_loss.backward()
        optimizer.step()

        summary_writer.add_scalar('Batch Loss/L1', training_l1_meter.item_average, epoch * len(train_loader) + i)
        summary_writer.add_scalar('Batch Loss/Huber', training_huber_meter.item_average, epoch * len(train_loader) + i)
        summary_writer.add_scalar('Batch Loss/L1-inv', training_l1_inv_meter.item_average, epoch * len(train_loader) + i)
        summary_writer.add_scalar('Batch Loss/L1-rel', training_l1_rel_meter.item_average, epoch * len(train_loader) + i)
        info_printer.set_description_str(info.format(training_l1_meter, training_l1_inv_meter, training_l1_rel_meter, training_huber_meter))

    if Config.train_validate:
        validation_l1_loss, validation_huber_loss, validation_l1_inv_loss, validation_l1_rel_loss = validate(val_loader=val_loader,
                                                                                                             model=model,
                                                                                                             forward_pass_function=forward_pass_function)

        summary_writer.add_scalar('L1 Loss/Training', training_l1_meter.avg, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('L1 Loss/Validation', validation_l1_loss, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('Huber Loss/Training', training_huber_meter.avg, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('Huber Loss/Validation', validation_huber_loss, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('L1-inv Loss/Training', training_l1_inv_meter.avg, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('L1-inv Loss/Validation', validation_l1_inv_loss, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('L1-rel Loss/Training', training_l1_rel_meter.avg, (epoch + 1) * len(train_loader))
        summary_writer.add_scalar('L1-rel Loss/Validation', validation_l1_rel_loss, (epoch + 1) * len(train_loader))

        if validation_l1_loss < best_loss[0] or validation_huber_loss < best_loss[1] or \
                validation_l1_inv_loss < best_loss[2] or validation_l1_rel_loss < best_loss[3]:
            best_loss[0] = min(validation_l1_loss, best_loss[0])
            best_loss[1] = min(validation_huber_loss, best_loss[1])
            best_loss[2] = min(validation_l1_inv_loss, best_loss[2])
            best_loss[3] = min(validation_l1_rel_loss, best_loss[3])

            # save best checkpoint
            checkpoint_list = []
            for k, module in enumerate(model):
                entry = {
                    'name': "module_" + str(k),
                    'epoch': epoch + 1,
                    'state_dict': module.state_dict()
                }
                checkpoint_list.append(entry)

            save_checkpoint(run_directory,
                            checkpoint_list,
                            step=(epoch + 1) * len(train_loader),
                            loss=[validation_l1_loss, validation_l1_inv_loss, validation_l1_rel_loss, validation_huber_loss])

            save_optimizer(run_directory,
                           optimizer=optimizer,
                           step=(epoch + 1) * len(train_loader),
                           loss=[validation_l1_loss, validation_l1_inv_loss, validation_l1_rel_loss, validation_huber_loss])

        # switch back to train mode !!!
        switch_mode(model=model, mode='train')


def validate(val_loader, model, forward_pass_function):
    validation_l1_meter = LossMeter()
    validation_huber_meter = LossMeter()
    validation_l1_inv_meter = LossMeter()
    validation_l1_rel_meter = LossMeter()

    # switch to evaluate mode
    switch_mode(model=model, mode='eval')

    with torch.no_grad():
        for i, (images, depths, poses, K) in enumerate(tqdm(val_loader)):
            batch_l1_meter, batch_huber_meter, batch_l1_inv_meter, batch_l1_rel_meter, \
            optimizer_loss, predictions, predictions_names = forward_pass_function(images=images,
                                                                                   depths=depths,
                                                                                   poses=poses,
                                                                                   K=K,
                                                                                   model=model,
                                                                                   is_training=False)
            # record losses
            validation_l1_meter.update(loss=batch_l1_meter.sum, count=batch_l1_meter.count)
            validation_huber_meter.update(loss=batch_huber_meter.sum, count=batch_huber_meter.count)
            validation_l1_inv_meter.update(loss=batch_l1_inv_meter.sum, count=batch_l1_inv_meter.count)
            validation_l1_rel_meter.update(loss=batch_l1_rel_meter.sum, count=batch_l1_rel_meter.count)

    return validation_l1_meter.avg, validation_huber_meter.avg, validation_l1_inv_meter.avg, validation_l1_rel_meter.avg
