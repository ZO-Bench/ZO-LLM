# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/1/17
"""WANDA: Pruning by Weights and activations (https://arxiv.org/pdf/2306.11695.pdf)"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel


@dataclass
class WandaPruningConfig:
    sparsity: Optional[float] = 0.5
    comparison_group: Optional[str] = "model"  # model, layer, linear
    scaler_device: Optional[str] = "cuda"


class WandaPruner:
    def __init__(
            self,
            config: WandaPruningConfig,
            model: PreTrainedModel,
    ):
        self.model = model
        self.config = config

        self.linear_module_scaler_dict = {}
        self.linear_module_num_samples_dict = {}
        self.linear_module_hook_dict = {}
        self._init_hooks_and_scaler()

    def _init_hooks_and_scaler(self):
        device = self.config.scaler_device

        def _get_forward_hook(name, input_dim):

            def _forward_hook(module, input, output):
                input = input.reshape(-1, input_dim)
                num_prev_samples = self.linear_module_num_samples_dict[name]
                num_curr_samples = input.shape[0]
                self.linear_module_scaler_dict[name] *= (num_prev_samples / (num_prev_samples + num_curr_samples))
                self.linear_module_num_samples_dict[name] += num_curr_samples
                self.linear_module_scaler_dict[name] += torch.norm(
                    input, dim=0, p=2
                ) / (num_prev_samples + num_curr_samples)

            return _forward_hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.linear_module_num_samples_dict[name] = 0
                self.linear_module_scaler_dict[name] = torch.ones(module.weight.shape[1], device=device)
                _hook = module.register_forward_hook(_get_forward_hook(name, module.weight.shape[1]))
                self.linear_module_hook_dict[name] = _hook

    def remove_hooks(self):
        for name, hook in self.linear_module_hook_dict.items():
            hook.remove()
