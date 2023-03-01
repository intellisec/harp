import torch


def rate_act_func(k_score, k_min):
    k = torch.sigmoid(k_score)
    k = k * (1 - k_min)  # E.g. global_k = 0.1, Make layer k in range [0.0, 0.99]
    k = k + k_min  # Make layer k in range [0.01, 1.0]
    return k


def rate_init_func(k, k_min, device):
    inv_k = torch.tensor((k - k_min) / (1 - k_min), device=device)
    clip_dec = 1e-2  # if args.prune_reg == 'channel' else 1e-4

    # 1e-3 helps to avoid sigmoid(+-1.0) = +-inf = NaN
    inv_k = torch.clip(inv_k, min=-1.0 + clip_dec, max=1.0 - clip_dec)

    k_score_inv = torch.log(inv_k / (1 - inv_k))

    return k_score_inv
