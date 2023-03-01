# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import logging
import time
from pathlib import Path

from args import parse_args
from utils.model import display_loadrate, initialize_stg_rate
from utils.logging import parse_configs_file
from utils.adv import (
    pgd_whitebox,
    fgsm_whitebox,
    cw_whitebox,
    autopgd_whitebox,
    autoattack_whitebox
)
from utils.logging import (
    AverageMeter,
    ProgressMeter,
    parse_prune_stg
)

from sklearn.metrics import balanced_accuracy_score

args = parse_args()
parse_configs_file(args)

if args.exp_mode == 'pretrain' and args.evaluate:
    task_name = os.path.join(Path(args.result_dir), args.exp_name)
else:
    task_name = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)
args.source_net = f'{task_name}/latest_exp/checkpoint/model_best.pth.tar'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn

import models
import data as Data
from utils.schedules import get_lr_policy, get_optimizer

from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
)

ATTACK_LIST = ['fgsm', 'pgd10', 'cw', 'autopgd', 'autoattack']
RUN_BACC = False

# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning


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


def main():

    # sanity checks
    if args.exp_mode in ["score_prune", "score_finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode, 'adv_eval')

    if not os.path.exists(result_main_dir):
        os.makedirs(result_main_dir, exist_ok=True)
    # parse_prune_stg(args)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_main_dir, f"eval_advs.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    num_gpus = len(args.gpu.strip().split(","))
    # gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    gpu_list = [i for i in range(num_gpus)]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Dataloader
    D = Data.__dict__[args.dataset](args)
    train_loader, test_loader = D.data_loaders()

    # Create model
    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes, mean=D.mean, std=D.std,
                prune_reg=args.prune_reg, task_mode=args.exp_mode, normalize=args.normalize
            ),
            gpu_list,
        ).to(device)
    else:
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes, mean=D.mean, std=D.std,
            prune_reg=args.prune_reg, task_mode=args.exp_mode, normalize=args.normalize
        ).to(device)
    logger.info(model)

    if args.exp_mode in ["rate_prune", "harp_prune"]:
        parse_prune_stg(args)

    # Customize models for training/pruning/fine-tuning
    prepare_model(model, args)

    logger.info(
        f"Dataset: {args.dataset}, D: {D}, num_train: {len(train_loader.dataset)}, num_test:{len(test_loader.dataset)}")

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            # model.load_state_dict(
            #     checkpoint["state_dict"], strict=False
            # )  # allows loading dense models
            model_dict = model.state_dict()
            """
            if (args.arch == "resnet18" or args.arch == "ResNet18") and args.exp_mode == 'score_prune':
                checkpoint_dict = checkpoint['net']
            else:
                checkpoint_dict = checkpoint['state_dict']
            """
            checkpoint_dict = checkpoint['state_dict']
            if args.exp_mode in ['pretrain'] and args.evaluate:
                if args.dataset == 'imagenet' and args.gpu.find(',') == -1:
                    checkpoint_dict = {k.replace("module.", ""): v for k, v in checkpoint_dict.items()
                                       if k.find('popup_scores') == -1 and k.find('sub_block') == -1}
                else:
                    checkpoint_dict = {k.replace("module.basic_model.", ""): v for k, v in checkpoint_dict.items()
                                       if k.find('popup_scores') == -1 and k.find('sub_block') == -1}
                model_dict.update(checkpoint_dict)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(checkpoint_dict)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}' !!!".format(args.source_net))

    # # Init scores once source net is loaded.
    # # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
    # if args.scaled_score_init:
    #     initialize_scaled_score(model, args.prune_reg)
    #
    # if args.rate_stg_init:
    #     initialize_stg_rate(model, args, device, logger)

    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )
    # resume (if checkpoint provided). Continue training with preiovus settings.
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    display_loadrate(model, logger, args)

    # Run attack
    logger.info(f'>>Evaluate on {ATTACK_LIST}')
    for attack in ATTACK_LIST:
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
            len(test_loader),
            [batch_time, losses, adv_losses, top1, top5, adv_top1, adv_top5],
            prefix="Test: ",
        )

        # if args.dataset == 'imagenet':
        #     mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
        #     std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
        # else:
        #     mean = torch.Tensor(np.array([0.0, 0.0, 0.0])[:, np.newaxis, np.newaxis])
        #     std = torch.Tensor(np.array([1.0, 1.0, 1.0])[:, np.newaxis, np.newaxis])
        #
        # mean = mean.expand(3, args.image_dim, args.image_dim).cuda()
        # std = std.expand(3, args.image_dim, args.image_dim).cuda()

        # switch to evaluate mode
        model.eval()

        nat_labels = []
        nat_preds_all = []
        adv_preds_all = []

        end = time.time()

        with torch.no_grad():

            for i, data in enumerate(test_loader):

                images, target = data[0].to(device), data[1].to(device)

                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # # compute output
                # images = images - mean
                # images.div_(std)

                # Compute nat output
                output_nat = model(images)

                # adversarial images
                if attack == 'pgd10':
                    attacker = pgd_whitebox
                    args.num_steps = 10
                elif attack == 'pgd20':
                    attacker = pgd_whitebox
                    args.num_steps = 20
                elif attack == 'pgd50':
                    attacker = pgd_whitebox
                    args.num_steps = 50
                    args.step_size = 2.5*args.epsilon/args.num_steps
                elif attack == 'fgsm':
                    attacker = fgsm_whitebox
                elif attack == 'cw':
                    attacker = cw_whitebox
                    args.num_steps = 20
                elif attack == 'autopgd':
                    attacker = autopgd_whitebox
                    args.num_steps = 50
                elif attack == 'autoattack':
                    attacker = autoattack_whitebox
                else:
                    raise NameError(f'{attack} is not supported for white-box attack!')

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

                # measure accuracy and record loss
                _, nat_preds = output_nat.topk(1, 1, True, True)
                nat_preds = nat_preds.view(-1).cpu().numpy()
                nat_labels = np.append(nat_labels, target.cpu().numpy().squeeze())
                nat_preds_all = np.append(nat_preds_all, nat_preds)

                _, adv_preds = output_adv.topk(1, 1, True, True)
                adv_preds = adv_preds.view(-1).cpu().numpy()
                adv_preds_all = np.append(adv_preds_all, adv_preds)

                nat_acc1, nat_acc5 = accuracy(output_nat, target, topk=(1, 5))
                adv_acc1, adv_acc5 = accuracy(output_adv, target, topk=(1, 5))

                top1.update(nat_acc1[0], images.size(0))
                top5.update(nat_acc5[0], images.size(0))
                adv_top1.update(adv_acc1[0], images.size(0))
                adv_top5.update(adv_acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % args.print_freq == 0:
                    progress.display(i)

            progress.display(i)  # print final results

        top1_bacc = balanced_accuracy_score(nat_labels, nat_preds_all) * 100.0
        adv_top1_bacc = balanced_accuracy_score(nat_labels, adv_preds_all) * 100.0

        logger.info(
            f"STANDARD ACC: Benign validation accuracy: {top1.avg}, Adversarial validation accuracy by {attack.upper()}: {adv_top1.avg}")

        if args.dataset == 'SVHN':
            logger.info(
                f"BALANCED ACC: Benign validation accuracy: {top1_bacc}, Adversarial validation accuracy by {attack.upper()}: {adv_top1_bacc}")

if __name__ == "__main__":
    main()
