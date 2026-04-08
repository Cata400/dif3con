import torch
from torch.optim import Adam

class AdamWithEpochEMA(Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        # Run Adam's normal update
        loss = super().step(closure)

        # Update exp_avg_sq_epoch
        for group in self.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'exp_avg_sq_epoch' not in state:
                    state['exp_avg_sq_epoch'] = torch.zeros_like(p.data)

                grad = p.grad.data
                exp_avg_sq_epoch = state['exp_avg_sq_epoch']

                exp_avg_sq_epoch.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        return loss

    def reset_epoch_moments(self):
        """Reset the per-epoch exp_avg_sq_epoch buffer for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'exp_avg_sq_epoch' in state:
                    state['exp_avg_sq_epoch'].zero_()
