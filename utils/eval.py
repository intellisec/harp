import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

from utils.logging import AverageMeter, ProgressMeter
from utils.adv import pgd_whitebox, fgsm, fgsm_whitebox, cw_whitebox, cw_loss

import numpy as np
import time


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def base(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    if args.dataset == 'imagenet':
        mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
    else:
        mean = torch.Tensor(np.array([0.0, 0.0, 0.0])[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array([1.0, 1.0, 1.0])[:, np.newaxis, np.newaxis])

    mean = mean.expand(3, args.image_dim, args.image_dim).cuda()
    std = std.expand(3, args.image_dim, args.image_dim).cuda()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            images = images - mean
            images.div_(std)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return top1.avg, top5.avg


def adv(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluate on adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top5 = AverageMeter("Adv-Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top5, adv_top1, adv_top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # adversarial images
            if args.attack_eval == 'pgd':
                attacker = pgd_whitebox
            elif args.attack_eval == 'fgsm':
                attacker = fgsm_whitebox
            elif args.attack_eval == 'cw':
                attacker = cw_whitebox
            else:
                raise NameError(f'{args.attack_eval} is not supported for white-box attack!')

            images = attacker(
                model,
                images,
                target,
                device,
                args.epsilon,
                args.num_steps,
                args.step_size,
                args.clip_min,
                args.clip_max,
                is_random=not args.const_init,
            )

            # compute output
            output_adv = model(images)
            loss = criterion(output_adv, target)

            # measure accuracy and record loss
            adv_acc1, adv_acc5 = accuracy(output_adv, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(adv_acc1[0], images.size(0))
            adv_top5.update(adv_acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return top1.avg, top5.avg, adv_top1.avg, adv_top5.avg, losses.avg, adv_losses.avg


def freeadv(model, device, val_loader, criterion, args, writer, epoch=0, attack='pgd'):

    # assert (
    #     not args.normalize
    # ), "Explicit normalization is done in the training loop, Dataset should have [0, 1] dynamic range."

    # Mean/Std for normalization
    if args.dataset == 'imagenet':
        mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
    else:
        mean = torch.Tensor(np.array([0.0, 0.0, 0.0])[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array([1.0, 1.0, 1.0])[:, np.newaxis, np.newaxis])

    mean = mean.expand(3, args.image_dim, args.image_dim).cuda()
    std = std.expand(3, args.image_dim, args.image_dim).cuda()

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ",
    )

    eps = args.epsilon
    K = args.num_steps
    step = args.step_size
    model.eval()
    end = time.time()
    print(" PGD eps: {}, num-steps: {}, step-size: {} ".format(eps, K, step))
    for i, (input, target) in enumerate(val_loader):

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        orig_input = input.clone()
        if attack in ['pgd', 'cw']:
            randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).to(device)
            input += randn
            input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            if attack == 'cw':
                with torch.enable_grad():
                    ascend_loss = cw_loss(output, target, num_classes=1000)
            else:
                ascend_loss = criterion(output, target)
            # ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - eps, input)
            input = torch.min(orig_input + eps, input)
            input.clamp_(0, 1.0)

        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

        if writer:
            progress.write_to_tensorboard(writer, "test", epoch * len(val_loader) + i)

        # write a sample of test images to tensorboard (helpful for debugging)
        if i == 0 and writer:
            writer.add_image(
                "Adv-test-images",
                torchvision.utils.make_grid(input[0 : len(input) // 4]),
            )

    progress.display(i)  # print final results

    return top1.avg, top5.avg, top1.avg, top5.avg, losses.avg, losses.avg


def nat_imagenet(model, args, device, test_loader):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
    std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, args.image_dim, args.image_dim).cuda()
    std = std.expand(3, args.image_dim, args.image_dim).cuda()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        for i, data in enumerate(test_loader):

            images, target = data[0].to(device), data[1].to(device)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            images = images - mean
            images.div_(std)

            # Compute nat output
            output_nat = model(images)

            nat_acc1, nat_acc5 = accuracy(output_nat, target, topk=(1, 5))

            top1.update(nat_acc1[0], images.size(0))
            top5.update(nat_acc5[0], images.size(0))

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return top1.avg, top5.avg
