import math
import warnings

import torch
import torch.nn as nn
from src.models.cpd_models import CPDModel
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset


class ABAnnealingLR(_LRScheduler):
    """Step size scheduler for SGLD.

    a and b are computed based on start and final step size.

    .. math::
      \epsilon_t = a(b + t)^{-\gamma}

    .. _SGLD\: Bayesian Learning via Stochastic Gradient Langevin Dynamics:
          https://icml.cc/2011/papers/398_icmlpaper.pdf
    """

    def __init__(self, optimizer, final_lr, gamma, T_max, last_epoch=-1, verbose=False):
        self.final_lr = final_lr
        self.gamma = gamma
        self.T_max = T_max

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs

        new_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if self.last_epoch > self.T_max:
                new_lrs.append(group["lr"])
            else:
                b = self.T_max / (
                    (base_lr / self.final_lr) * math.exp(1 / self.gamma) - 1.0
                )
                a = base_lr * b**self.gamma

                new_lr = a / (b + self.last_epoch) ** self.gamma
                new_lrs.append(new_lr)

        return new_lrs


class CustomNoisyAdam(Adam):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, temperature=0.0
    ):
        super().__init__(params, lr=lr, betas=betas)
        self.weight_decay = weight_decay
        self.temperature = temperature

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                _ = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if self.weight_decay != 0:
                    grad = grad.add(p.data, alpha=self.weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # add noise
                noise = torch.rand_like(exp_avg)
                p.data.add_(noise, alpha=math.sqrt(2 * group["lr"] * self.temperature))


class CPDModelCustomNoisyAdam(CPDModel):
    def __init__(
        self,
        loss_type: str,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> None:
        super().__init__(loss_type, args, model, train_dataset, test_dataset)

        self.temperature = args["learning"]["temperature"]

        self.T_max = args["learning"]["T_max"]

        self.final_lr = args["learning"]["final_lr"]
        self.gamma = args["learning"]["gamma"]

    def configure_optimizers(self):
        optimizer = CustomNoisyAdam(
            self.model.parameters(), lr=self.learning_rate, temperature=self.temperature
        )

        scheduler = ABAnnealingLR(
            optimizer,
            final_lr=self.final_lr,
            gamma=self.gamma,
            T_max=self.T_max,
            verbose=False,
        )

        return [optimizer], [scheduler]
