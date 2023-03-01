import torch
import json
import numpy as np
import logging
import shutil
import os
import yaml
import sys
from distutils.dir_util import copy_tree
from utils.model import subnet_to_dense


def save_checkpoint(
    state, is_best, args, result_dir, filename="checkpoint.pth.tar", save_dense=False
):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )

    if save_dense:
        state["state_dict"] = subnet_to_dense(state["state_dict"], args.k)
        torch.save(
            subnet_to_dense(state, args.k),
            os.path.join(result_dir, "checkpoint_dense.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(result_dir, "checkpoint_dense.pth.tar"),
                os.path.join(result_dir, "model_best_dense.pth.tar"),
            )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst, verbose=False)


# ref:https://github.com/allenai/hidden-networks/blob/master/configs/parser.py
def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


# ref: https://github.com/allenai/hidden-networks/blob/master/args.py
def parse_configs_file(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    config_file = f"configs/config-{args.dataset.lower()}.yml"

    # load yaml file
    yaml_txt = open(config_file).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {config_file}")
    args.__dict__.update({'configs': config_file})
    args.__dict__.update(loaded_yaml)

    "Define result dir"
    args.__dict__.update({'result_dir': f"./trained_models/{args.arch}"})

    # Compose exp_name
    if args.exp_mode != 'pretrain':
        exp_name = '_'.join([args.dataset.lower(), args.prune_reg, args.stg_id, str(args.k)])
    if args.exp_mode == 'pretrain':
        exp_name = 'pretrain'
    args.__dict__.update({'exp_name': exp_name})

    # Adpat scaled_score_init switch:
    if args.exp_mode in ['score_prune', 'rate_prune', 'harp_prune']:
        args.__dict__.update({'scaled_score_init': True})
    if args.exp_mode in ['score_finetune', "harp_finetune"]:
        args.__dict__.update({'scaled_score_init': False})

    # Adapt rate_stg_init switch:
    if args.exp_mode in ['rate_prune', 'harp_prune'] or (args.exp_mode == 'pretrain' and args.evaluate):
        args.__dict__.update({'rate_stg_init': True})
    else:
        args.__dict__.update({'rate_stg_init': False})

    if args.exp_mode in ['rate_prune', 'harp_prune']:
        args.__dict__.update({'soft_hw': True})
    else:
        args.__dict__.update({'soft_hw': False})

    # Define beta for adv train
    if args.adv_loss == 'trades':
        args.__dict__.update({'beta': 6.0})
    if args.adv_loss == 'mart':
        args.__dict__.update({'beta': 5.0})

    # Learning rate for prune and finetune stages
    if args.exp_mode.find('prune') != -1:
        args.__dict__.update({'lr': 0.1})
    if args.exp_mode.find('finetune') != -1:
        args.__dict__.update({'lr': 0.01})

    # Define root directory for individual experiment
    preprocess = 'normalize' if args.normalize else ''
    args.result_dir = os.path.join(args.result_dir, args.dataset, preprocess, args.adv_loss)

    # Define pretrained source net for pruning
    if args.exp_mode in ['rate_prune', 'harp_prune', 'score_prune']:
        source_net = os.path.join(args.result_dir, 'pretrain', 'latest_exp/checkpoint/model_best.pth.tar')
        args.__dict__.update({'source_net': source_net})
        print(f'Source_net: {args.source_net}')

    if args.exp_mode == 'rate_finetune':
        if not args.evaluate:
            source_net = os.path.join(args.result_dir, args.exp_name, 'rate_prune', 'latest_exp/checkpoint/model_best.pth.tar')
            args.__dict__.update({'source_net': source_net})
        print(f'Source_net: {args.source_net}')

    if args.exp_mode == 'harp_finetune':
        if not args.evaluate:
            source_net = os.path.join(args.result_dir, args.exp_name, 'harp_prune', 'latest_exp/checkpoint/model_best.pth.tar')
            args.__dict__.update({'source_net': source_net})
        print(f'Source_net: {args.source_net}')

    if args.exp_mode == 'score_finetune':
        if not args.evaluate:
            source_net = os.path.join(args.result_dir, args.exp_name, 'score_prune', 'latest_exp/checkpoint/model_best.pth.tar')
            args.__dict__.update({'source_net': source_net})
        print(f'Source_net: {args.source_net}')

    if args.exp_mode == 'pretrain' and args.evaluate:
        source_net = os.path.join(args.result_dir, args.exp_name, 'latest_exp/checkpoint/model_best.pth.tar')
        args.__dict__.update({'source_net': source_net})
        print(f'Source_net: {args.source_net}')


def parse_prune_stg(args):
    strategy_f = 'strategies.json'
    strategies = json.load(open(os.path.join('configs', strategy_f)))

    harp_stg = strategies[args.arch][args.prune_reg][args.stg_id]
    prune_stg = [[], [], []]
    if args.arch == 'vgg16_bn':
        assert len(harp_stg) == 16, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
        prune_stg[0] = harp_stg[:13]
        prune_stg[1] = harp_stg[13:]
        prune_stg[2] = []
    elif args.arch == 'resnet18':
        if args.prune_reg == 'channel':
            assert len(harp_stg) == 18, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = harp_stg[:17]
            prune_stg[1] = harp_stg[17:]
            prune_stg[2] = list(np.take(harp_stg, (5, 9, 13)))
        if args.prune_reg == 'weight':
            assert len(harp_stg) == 21, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = harp_stg[:7] + harp_stg[8:12] + harp_stg[13:17] + harp_stg[18:20]
            prune_stg[1] = harp_stg[20:]
            prune_stg[2] = list(np.take(harp_stg, (7, 12, 17)))
    elif args.arch == 'wrn_28_4':
        if args.prune_reg == 'channel':
            assert len(harp_stg) == 26, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = harp_stg[:25]
            prune_stg[1] = harp_stg[25:]
            prune_stg[2] = list(np.take(harp_stg, (1, 9, 17)))
        if args.prune_reg == 'weight':
            assert len(harp_stg) == 29, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = harp_stg[:3] + harp_stg[4:12] + harp_stg[13:21] + harp_stg[22:28]
            prune_stg[1] = harp_stg[28:]
            prune_stg[2] = list(np.take(harp_stg, (3, 12, 21)))
    elif args.arch == 'ResNet50':
        if args.prune_reg == 'channel':
            assert len(harp_stg) == 50, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = harp_stg[:49]
            prune_stg[1] = harp_stg[49:]
            prune_stg[2] = list(np.take(harp_stg, (1, 10, 22, 40)))
        if args.prune_reg == 'weight':
            assert len(harp_stg) == 54, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = harp_stg[:4] + harp_stg[5:14] + harp_stg[15:27] + harp_stg[28:46] + harp_stg[47:53]
            prune_stg[1] = harp_stg[53:]
            prune_stg[2] = list(np.take(harp_stg, (4, 14, 27, 46)))
    else:
        raise NameError('Strategy check only supports vgg16_bn, resnet18, wrn_28_4, resnet50 but no "{}"'.format(args.arch))

    args.__dict__.update({'conv_k': prune_stg[0],
                          'fc_k': prune_stg[1],
                          'shortcut_k': prune_stg[2]})

    # if args.exp_mode.find('harp') != -1:
    log_outdir = os.path.join(args.result_dir, args.exp_name)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(log_outdir, "load_init_strategy.log"), "a")
    )

    logger.info(
        '>> {} prune with target rate {} on model {}'.format(args.prune_reg.upper(), args.k, args.arch.upper()))
    logger.info('$  Prune-Strategy on SubnetConv:   {}'.format(prune_stg[0]))
    logger.info('$  Prune-Strategy on SubnetLinear: {}'.format(prune_stg[1]))
    logger.info('$  Prune-Strategy on ShortcutConv: {}'.format(prune_stg[2]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)

