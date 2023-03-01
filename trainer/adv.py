import time
import importlib
import torch
import torch.nn as nn
import torchvision
import math

from utils.logging import AverageMeter, ProgressMeter
from utils.eval import accuracy
from utils.hw import hw_loss, hw_flops_loss
from utils.model import map_shortcut_rate
from utils.utils import rate_act_func


# TODO: add adversarial accuracy.
def train(
    model, device, train_loader, criterion, optimizer, epoch, args, writer, frozen_gamma
):
    warmup_epochs = args.warmup_epochs
    if epoch < warmup_epochs:
        print(
            " ->->->->->->->->->-> One epoch with Nat Warm-Up [Warmup Epoch: {}] <-<-<-<-<-<-<-<-<-<-<-<-<-<-<-".format(epoch)
        )
    else:
        print(
            " ->->->->->->->->->-> One epoch with Adversarial training [AT Epoch: {}] <-<-<-<-<-<-<-<-<-<-".format(epoch-warmup_epochs)
        )

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    hw_losses = AverageMeter("HW-Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    info_list = [batch_time, losses, top1, top5] if args.exp_mode == 'pretrain' else [batch_time, losses, hw_losses, top1, top5]
    progress = ProgressMeter(
        len(train_loader),
        info_list,
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    dataloader = train_loader

    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training data
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(f"Training images range: {[torch.min(images), torch.max(images)]}")

        output = model(images)

        # calculate robust loss
        if epoch < warmup_epochs:
            adv_loss = getattr(importlib.import_module("utils.adv"), args.warmup_loss+'_loss')
        else:
            adv_loss = getattr(importlib.import_module("utils.adv"), args.adv_loss+'_loss')

        loss = adv_loss(
            model=model,
            x_natural=images,
            y=target,
            device=device,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            distance=args.distance,
        )

        if args.soft_hw:
            if args.prune_reg == 'channel':
                hw_loss_func = hw_flops_loss
            else:
                hw_loss_func = hw_loss
            gamma, loss_hw, _ = hw_loss_func(
                model=model,
                device=device,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                frozen_gamma=frozen_gamma
            )
            hw_losses.update(loss_hw.item(), images.size(0))

            loss = loss + gamma * loss_hw

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
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

        # write a sample of training images to tensorboard (helpful for debugging)
        if i == 0:
            writer.add_image(
                "training-images",
                torchvision.utils.make_grid(images[0 : len(images) // 4]),
            )

    for m_name, m in model.named_modules():
        if hasattr(m, "k_rate"):
            k = rate_act_func(m.k_score.data, m.k_min)
            print(f'{m_name}: {k.data}')
