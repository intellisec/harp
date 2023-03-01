import torch
import torch.nn as nn

import os
import math
import numpy as np
from utils.hw import vgg16_bn_flops, resnet18_flops
from models.layers import SubnetConv, SubnetLinear
from utils.utils import rate_act_func, rate_init_func

# TODO: avoid freezing bn_params
# Some utils are borrowed from https://github.com/allenai/hidden-networks and https://github.com/inspire-group/hydra
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores", "k_score"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name, prune_reg='weight'):
    assert var_name in ["weight", "bias", "popup_scores", "k_score"]
    for v_n, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                if v_n.lower().find('shortcut') != -1 or v_n.lower().find('downsample') != -1:
                    if prune_reg == 'channel' and var_name == 'k_score':
                        getattr(v, var_name).requires_grad = False
                    else:
                        getattr(v, var_name).requires_grad = True
                else:
                    getattr(v, var_name).requires_grad = True


def set_prune_rate_model(model, args, device):
    for block_n, v in model.named_modules():

        if hasattr(v, "set_prune_rate"):

            if args.exp_mode in ["harp_prune", "rate_prune"]:
                if block_n.find('sub_block') != -1:
                    # Actually sub_block doesn't exit in forward(), here is just as placeholder for no-bug
                    v.set_prune_rate(args.k, args.k, args.alpha, device)
                else:
                    v_name = v._get_name()
                    if block_n.lower().find('shortcut') != -1 or block_n.lower().find('downsample') != -1:
                        k = args.shortcut_k.pop(0)
                    elif v_name == 'SubnetConv':
                        k = args.conv_k.pop(0)
                    elif v_name == 'SubnetLinear':
                        k = args.fc_k.pop(0)
                    else:
                        raise NameError('{} has no pruning rate!'.format(v_name))

                    print('$$$ {} prune rate {:.8f} on {}'.format(args.prune_reg, k, v))
                    v.set_prune_rate(k, args.k, args.alpha, device)
            else:
                v.set_prune_rate(args.k, args.k, args.alpha, device)


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "subnet":
        return SubnetConv, SubnetLinear
    else:
        raise ValueError("Incorrect layer type")


def show_gradients(model, logger):
    for i, v in model.named_parameters():
        logger.info(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def initialize_scaled_score(model, prune_reg='weight'):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES) | Prune_Reg: {}".format(prune_reg)
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")

            if prune_reg == 'weight':
                # Weight Pruning
                # """
                # Close to kaiming unifrom init
                m.popup_scores.data = (
                    math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
                )
                # """

            elif prune_reg == 'channel':
                # Channel Prune
                # """
                reshaped_weights = torch.sum(torch.abs(m.weight.data.reshape(m.weight.data.shape[1],-1)),dim=1)
                if type(m) == SubnetConv:
                    channel_popup_scores = (math.sqrt(6/n)*reshaped_weights / torch.max(torch.abs(reshaped_weights))).reshape(1,m.weight.data.shape[1],1,1)
                else:
                    channel_popup_scores = (
                            math.sqrt(6 / n) * reshaped_weights / torch.max(torch.abs(reshaped_weights))).reshape(
                        1, m.weight.data.shape[1])
                m.popup_scores.data = channel_popup_scores
                # """
            else:
                raise NameError('Please check prune_reg, current "{}" is not in [weight, channel] !'.format(prune_reg))


def initialize_stg_rate(model, args, device, logger):
    logger.info(f'Initializing layer-wise rate with strategy: {args.stg_id}')

    for m_name, m in model.named_modules():
        if hasattr(m, "k_score"):
            if args.prune_reg == 'channel' and (m_name.find('shortcut') != -1 or m_name.find('downsample') != -1):
                pass
            else:
                # Update k_score
                m.k_score.data = rate_init_func(m.k, m.k_min, device)

                # Display real prune rate
                k = rate_act_func(m.k_score.data, m.k_min)

                logger.info(f'Initialize rate on {m_name}: {k}')


def display_loadrate(model, logger, args):
    ch_list = []
    orig_list = []
    logger.info(f"Load pruning strategy:")
    for m_name, m in model.named_modules():
        if hasattr(m, "k_score"):
            k = rate_act_func(m.k_score.data, m.k_min)
            # k = np.maximum(k, args.k*0.1)
            logger.info(f'{m_name}: {k}')

            if m.prune_reg == 'channel':
                orig_list.append(m.weight.shape[1])
                ch_list.append(round(m.weight.shape[1]*k.detach().cpu().numpy()))

    if args.prune_reg == "channel":
        model_name = model.module._get_name() if args.gpu.find(',') != -1 and not args.no_cuda else model._get_name()
        if model_name == 'VGG':
            flops_counter = vgg16_bn_flops
        elif model_name == 'ResNet':
            if model.num_classes == 10:
                flops_counter = resnet18_flops
        else:
            raise NameError(f'{model_name} has no FLOPs counter yet.')

        orig_flops = flops_counter(orig_list, 10)
        net_flops = flops_counter(ch_list, 10)
        logger.info(f"Original channel shape: {orig_list}")
        logger.info(f"Strategy after pruning: {ch_list}")
        logger.info(f"Network FLOPs: {net_flops}/{orig_flops}")
        logger.info(f"Compression rate in FLOPs = {round(net_flops/orig_flops, 3)}, xFLOPs = {round(orig_flops/net_flops, 3)}")


def map_shortcut_rate(model, args, verbose=False):

    assert args.prune_reg == 'channel', 'Shortcut rate mapping only support channel prune!'

    for m_name, m in model.named_modules():
        if hasattr(m, "shortcut") or hasattr(m, "convShortcut") or hasattr(m, 'downsample'):

            conv_rate = 0.0
            for l_n, l in m.named_modules():
                if conv_rate == 0.0 and hasattr(l, "k_score"):
                    conv_rate = l.k_score.data

                if l_n in ['shortcut', 'convShortcut', 'downsample']:

                    for v in l.modules():
                        if hasattr(v, "k_score"):
                            v.k_score.data = conv_rate

                            if verbose:
                                k = rate_act_func(v.k_score.data, v.k_min)
                                print(f'Mapping  rate on {m_name}.{l_n}: {k}')


def prepare_model(model, args, device='cpu'):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """

    # if args.exp_mode in ["score_prune", "harp_prune"]:
    set_prune_rate_model(model, args, device)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None  | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        freeze_vars(model, "k_score", args.freeze_bn)

    elif args.exp_mode == "score_prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")
        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)
        freeze_vars(model, "k_score", args.freeze_bn)

    elif args.exp_mode == "score_finetune":
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        freeze_vars(model, "k_score", args.freeze_bn)

    elif args.exp_mode == "rate_prune":
        print(f"#################### Tune layer rates & scores ####################")
        print(
            f"===>>  gradient for importance_scores and layer_scores"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)
        unfreeze_vars(model, "k_score", args.prune_reg)

    elif args.exp_mode == "rate_finetune":
        print(f"#################### Directly finetune network ####################")
        print(
            f"===>>  gradient for weights and bias"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        freeze_vars(model, "k_score", args.freeze_bn)

    elif args.exp_mode == "harp_prune":
        print(f"#################### Tune layer rates & scores ####################")
        print(
            f"===>>  gradient for importance_scores and layer_scores"
        )
        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)
        unfreeze_vars(model, "k_score", args.prune_reg)

    elif args.exp_mode == "harp_finetune":
        print(f"#################### Directly finetune network ####################")
        print(
            f"===>>  gradient for weights and bias"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        freeze_vars(model, "k_score", args.freeze_bn)
    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scores(model, args.scores_init_type)


def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            dense[k.replace("popup_scores", "weight")] = (
                subnet_dict[k.replace("popup_scores", "weight")] * out
            )
    return dense

def current_model_pruned_fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense models
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []
    zero_weights = 0
    total_params = 0

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    """
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    """
                    zero_weights += np.sum(d==0)
                    total_params += np.size(d)
                    # if verbose:
                    #     print(i, v, p)
        # return np.mean(pl)
        return (float(zero_weights)/float(total_params))*100.
        

def sanity_check_paramter_updates(model, last_ckpt):
    """
        Check whether weigths/popup_scores gets updated or not compared to last ckpt.
        ONLY does it for 1 layer (to avoid computational overhead)
    """
    for i, v in model.named_modules():
        if hasattr(v, "weight") and hasattr(v, "popup_scores"):
            if getattr(v, "weight") is not None:
                w1 = getattr(v, "weight").data.cpu()
                w2 = last_ckpt[i + ".weight"].data.cpu()
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
            if getattr(v, "k_score") is not None:
                r1 = getattr(v, "k_score").data.cpu()
                r2 = last_ckpt[i + ".k_score"].data.cpu()
            return not torch.allclose(w1, w2), not torch.allclose(s1, s2), not torch.allclose(r1, r2)
