import time
import numpy as np
import sys

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from utils.hw import hw_loss, hw_flops_loss
from utils.logging import AverageMeter, ProgressMeter
from utils.eval import accuracy
from utils.adv import fgsm
from utils.model import map_shortcut_rate


def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    epoch,
    args,
    writer,
    frozen_gamma
):

    # assert (
    #     not args.normalize
    # ), "Explicit normalization is done in the training loop, Dataset should have [0, 1] dynamic range."

    global_noise_data = torch.zeros(
        [args.batch_size, 3, args.image_dim, args.image_dim]
    ).to(device)

    mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, args.image_dim, args.image_dim).to(device)
    std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, args.image_dim, args.image_dim).to(device)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    hw_losses = AverageMeter("HW-Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, hw_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        data_time.update(time.time() - end)

        for _ in range(args.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(
                global_noise_data[0 : input.size(0)], requires_grad=True
            ).to(device)
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)

            if not args.normalize:
                in1.sub_(mean).div_(std)

            output = model(in1)
            loss = criterion(output, target)

            if args.soft_hw:
                gamma, loss_hw, _ = hw_loss(
                    model=model,
                    device=device,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    frozen_gamma=frozen_gamma
                )
                hw_losses.update(loss_hw.item(), input.size(0))

                loss = loss + gamma * loss_hw

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, args.epsilon)
            global_noise_data[0 : input.size(0)] += pert.data
            global_noise_data.clamp_(-args.epsilon, args.epsilon)

            optimizer.step()

            # Map shortcut layer rates for channel prune:
            if args.prune_reg == 'channel':
                map_shortcut_rate(model, args)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                progress.write_to_tensorboard(
                    writer, "train", epoch * len(train_loader) + i
                )

        if i == 0:
            print(
                in1.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(f"Training images range: {[torch.min(in1), torch.max(in1)]}")

        # write a sample of training images to tensorboard (helpful for debugging)
        if i == 0:
            writer.add_image(
                "training-images",
                torchvision.utils.make_grid(input[0 : len(input) // 4]),
            )
