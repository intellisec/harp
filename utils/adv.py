import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchattacks
from models.layers import SubnetConv, SubnetLinear
from utils.flops import count_flops_conv, count_flops_dense


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


# ref: https://github.com/yaodongyu/TRADES
def trades_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
        x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
    if distance == "l_inf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), F.softmax(model(x_natural), dim=1)
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return loss


def mart_loss(
        model,
        x_natural,
        y,
        device,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        distance='l_inf',
    ):

    kl = nn.KLDivLoss(reduction='none')

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss_mart = loss_adv + float(beta) * loss_robust

    return loss_mart


def cw_loss(output, target, confidence=50):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    num_classes = output.shape[1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def pgd_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
):
    model.eval()
    # generate adversarial example
    random_noise = (
        torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device)
    )
    x_adv = x_natural.detach() + random_noise
    if distance == "l_inf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.cross_entropy(model(x_adv), y, size_average=False)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss_pgd = natural_criterion(logits, y)
    return loss_pgd


def fgsm_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
):
    model.eval()
    # generate adversarial example
    # random_noise = (
    #     torch.FloatTensor(x_natural.shape).uniform_(-step_size, step_size).to(device)
    # )
    # x_adv = x_natural.detach() + random_noise
    x_adv = x_natural.detach()

    x_adv.requires_grad_()
    with torch.enable_grad():
        loss = F.cross_entropy(model(x_adv), y, size_average=False)
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    x_adv = torch.min(
        torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
    )
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss_fgsm = natural_criterion(logits, y)

    return loss_fgsm


def nat_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
):
    model.train()
    x_nat = Variable(x_natural, requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_nat)
    loss_nat = natural_criterion(logits, y)

    return loss_nat


# TODO: support L-2 attacks too.
def pgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
):

    x_adv = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_adv.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
        loss.backward()
        eta = step_size * x_adv.grad.data.sign()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
        x_adv = Variable(x.data + eta, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=True)

    return x_adv


def cw_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
):

    x_adv = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_adv.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = cw_loss(model(x_adv), y)
        loss.backward()
        eta = step_size * x_adv.grad.data.sign()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
        x_adv = Variable(x.data + eta, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=True)

    return x_adv


def fgsm_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
):

    x_adv = Variable(x.data, requires_grad=True)

    opt = optim.SGD([x_adv], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    eta = epsilon * x_adv.grad.data.sign()
    x_adv = Variable(x_adv.data + eta, requires_grad=True)
    eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
    x_adv = Variable(x.data + eta, requires_grad=True)
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=True)

    return x_adv


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def autopgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True
 ):
    attacker = torchattacks.APGD(model=model, eps=epsilon, loss='ce', steps=num_steps, n_restarts=int(num_steps/10))
    x_adv = attacker(x, y)
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    return x_adv


def autoattack_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True
 ):
    attacker = torchattacks.AutoAttack(model=model, eps=epsilon)
    x_adv = attacker(x, y)
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    return x_adv