# copy dependencies from transformers/optimization.py
import os
import math
import copy
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
import numpy as np
from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor
import torch.optim as optim

import json

import torch
import math


class GaussianFunction:
    def __init__(self, step, duration, total_steps):
        """
        Initialize the Gaussian function parameters.

        Args:
            duration (int): The range around the center where values are close to 1 (>0.6).
        """
        self.mu = step
        self.sigma = duration // 2
        self.total_steps = total_steps
        self.steps = torch.arange(self.total_steps + 1, dtype=torch.bfloat16)
        # Gaussian formula: exp(-((x - mean)^2 / (2 * std^2)))
        self.gaussian_values = torch.exp(-((self.steps - self.mu) ** 2) / (2 * self.sigma ** 2))


    def get_value(self, t):
        """
        Get the value of the Gaussian at a specific step t.

        Args:
            t (int): The step at which to compute the Gaussian value.

        Returns:
            float: The value of the Gaussian at step t.
        """
        return self.gaussian_values[t]

# # for debugging
# import matplotlib.pyplot as plt
# gaussian = GaussianFunction(step=200, duration=10, total_steps=10000)
# plt.plot(gaussian.gaussian_values.numpy())
# plt.title("Gaussian Function"), plt.xlabel("Steps"), plt.ylabel("Value"), plt.grid(True)
# plt.show()

class AdamWGradientInjection(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
            gap=100,
            name=None,
            log_folder=None,
            grad_injection_step=None,
            grad_injection_multiplier=None,
            grad_injection_elements=None,
            grad_injection_layer_number=None,
            grad_injection_fn=None,
            grad_injection_duration=None,
            total_steps=None,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        self.gap = gap
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

        # for gradient spike detection
        # self.current_step=0
        self.total_step = 0
        self.grad_dict = {}
        self.moment_dict = {}
        self.name = name
        self.moment_second_dict = {}
        self.log_folder = log_folder
        # for gradient injection

        if grad_injection_step:
            self.grad_injection = {
                # TODO: fix to work with multiple values
                'step': grad_injection_step[0],
                'multiplier': grad_injection_multiplier[0],
                'elements': grad_injection_elements[0],
                'layer_number': grad_injection_layer_number[0],
                'fn': grad_injection_fn,
                'duration': grad_injection_duration[0],
                'total_steps': total_steps,
            }

            if grad_injection_fn == "gaussian":
                self.gauss_values = GaussianFunction(step=self.grad_injection["step"],
                                                     duration=self.grad_injection["duration"],
                                                     total_steps=total_steps)


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # for gradient spike detection
        # self.current_step+=1
        self.total_step += 1
        layer_n = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if 'dim' not in group:
                    group['dim'] = 2
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # for injecting gradient
                # if layer_n in self.grad_injection['layer_number']:
                # if layer_n == self.grad_injection['layer_number']:
                condition_saving_gradients = len(grad.shape) > 1 and "embed" not in group["name"] and "head" not in group["name"]
                if condition_saving_gradients: # inject to all layers with at least 2 dims


                    if layer_n not in self.grad_injection:
                        self.grad_injection[layer_n] = {
                            'i': (torch.rand(self.grad_injection['elements']) * grad.shape[0]).to(torch.int),
                            'j': (torch.rand(self.grad_injection['elements']) * grad.shape[1]).to(torch.int),
                        }
                    gauss_mult = self.gauss_values.get_value(state["step"])
                    if gauss_mult > 1e-4:
                        grad_injection_multiplier = self.grad_injection['multiplier'] * gauss_mult
                        # grad_injection_elements = self.grad_injection['elements']
                        grad_injection = torch.ones_like(grad)
                        # grad_idxs_injection_i = (torch.rand(grad_injection_elements) * grad.shape[0]).to(torch.int32)
                        # grad_idxs_injection_j = (torch.rand(grad_injection_elements) * grad.shape[1]).to(torch.int32)
                        grad_idxs_injection_i = self.grad_injection[layer_n]['i']
                        grad_idxs_injection_j = self.grad_injection[layer_n]['j']
                        grad_injection[grad_idxs_injection_i, grad_idxs_injection_j] = grad_injection_multiplier
                        # inject
                        grad = grad * grad_injection
                        print(f"Gradient injected!\t"
                              f"(Elements: {self.grad_injection['elements']}, ",
                              f"Gauss mult: {gauss_mult:.4f}, "
                              f"Mult: {self.grad_injection['multiplier']}, "
                              f"Gauss*Mult: {gauss_mult * self.grad_injection['multiplier']:.4f})")



                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # INI - TIANJING CODE
                # # for gradient spike detection
                # # if 'rank' in group:
                # if n == 20:
                #     if n not in self.grad_dict.keys():
                #         print("save n", n)
                #         self.grad_dict[n] = []
                #         self.moment_dict[n] = []
                #         self.moment_second_dict[n] = []
                #     self.grad_dict[n].append(grad.detach().cpu())
                # # self.moment_dict[n].append(exp_avg.detach().cpu())
                # # self.moment_second_dict[n].append(exp_avg_sq.detach().cpu())
                # n += 1
                # END - TIANJING CODE

                # for gradient spike detection
                if condition_saving_gradients:
                # if layer_n == self.grad_injection['layer_number']:
                    if layer_n not in self.grad_dict.keys():
                        p_name = group["name"]
                        print(f"Save gradients for layer:\t{layer_n} ({p_name})")
                        self.grad_dict[layer_n] = []
                    self.grad_dict[layer_n].append(grad.detach().cpu().to(dtype=torch.float16))
                layer_n += 1

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                # if "rank" in group:
                #     norm_grad = state["projector"].project_back(norm_grad)
                # if "rank" in group:
                #     mask=torch.rand_like(norm_grad).to(norm_grad.device)>group["rank"]
                #     norm_grad[mask]=0
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        # for gradient spike detection
        if state['step'] % 50 == 0 and state['step'] < 1005:
        # if state['step'] % 10 == 0 and state['step'] < 11:
            # np.save("./grad_id.npy",self.trajectories_id)
            grad_dict = {str(key): torch.stack(value).float().numpy().astype(np.float16)
                         for key, value in self.grad_dict.items()}
            # moment_dict = {str(key): torch.stack(value).float().numpy() for key, value in self.moment_dict.items()}
            # moment_second_dict = {str(key): torch.stack(value).float().numpy() for key, value in self.moment_second_dict.items()}
            print("Saving gradients, don't stop the execution...")
            gradient_path = os.path.join(self.log_folder, self.name + "_grad_dict.npz")
            np.savez_compressed(gradient_path, **grad_dict)
            print("Saved at", gradient_path)

            # log grad injection params
            grad_info = copy.deepcopy(self.grad_injection)
            for k, v in grad_info.items():
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        if isinstance(v1, torch.Tensor):
                            grad_info[k][k1] = grad_info[k][k1].__repr__()
            with open(os.path.join(self.log_folder, self.name + "_grad_injection.json"), "w") as f:
                f.write(json.dumps(grad_info, indent=4))

        return loss
