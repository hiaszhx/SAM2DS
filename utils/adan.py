# utils/adan.py
import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List

class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
    """
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0, no_prox=False, foreach: bool = True):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm, no_prox=no_prox, foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)
            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())
            global_grad_norm = torch.sqrt(global_grad_norm)
            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            pre_grads = []

            beta1, beta2, beta3 = group['betas']
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'pre_grad' not in state or group['step'] == 1:
                    state['pre_grad'] = p.grad

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                pre_grads.append(state['pre_grad'])

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                pre_grads=pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                clip_global_grad_norm=clip_global_grad_norm,
            )
            
            # Use simple implementation for compatibility
            _single_tensor_adan(**kwargs)

            for p, copy_grad in zip(params_with_grad, grads):
                 # In single tensor impl, grads are modified in place or cloned, 
                 # but we need to update pre_grad in state
                 # The _single_tensor_adan above actually calculates update and applies to param
                 # We just need to store current grad as pre_grad for next step
                 self.state[p]['pre_grad'] = p.grad.clone()

        return loss

def _single_tensor_adan(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        exp_avg_diffs: List[Tensor],
        pre_grads: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        bias_correction3_sqrt: float,
        lr: float,
        weight_decay: float,
        eps: float,
        no_prox: bool,
        clip_global_grad_norm: Tensor,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        pre_grad = pre_grads[i]

        if clip_global_grad_norm < 1.0:
            grad.mul_(clip_global_grad_norm)

        diff = grad - pre_grad
        update = grad + beta2 * diff

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)
        exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)

        denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
        step_update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2).div_(denom)

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.add_(step_update, alpha=-lr)
        else:
            param.add_(step_update, alpha=-lr)
            param.div_(1 + lr * weight_decay)