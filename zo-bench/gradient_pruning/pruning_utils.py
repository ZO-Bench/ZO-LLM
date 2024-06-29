# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/1/18
from typing import Dict

import torch
from transformers import PreTrainedModel


@torch.no_grad()
def random_mask_like(tensor, nonzero_ratio, generator=None):
    """
    Return a random mask with the same shape as the input tensor, where the fraction of True is equal to the sparsity.

    Examples
    --------
    >>> random_mask_like(torch.randn(10, 10), 0.1).count_nonzero()
    tensor(10)
    """
    mask = torch.zeros_like(tensor)
    mask.view(-1)[torch.randperm(mask.numel(), generator=generator)[:int(nonzero_ratio * mask.numel())]] = 1
    return mask.bool()


@torch.no_grad()
def fast_random_mask_like(tensor, nonzero_ratio, generator=None):
    """
    A much faster version of random_zero_mask_like, but the sparsity is not guaranteed.

    Examples
    --------
    >>> fast_random_mask_like(torch.randn(10, 10), 0.1).count_nonzero() < 20
    tensor(True)
    """
    mask = torch.empty_like(tensor).normal_(generator=generator) < nonzero_ratio
    return mask.bool()


@torch.no_grad()
def estimate_pretrained_model_magnitude_pruning_threshold(
        model: PreTrainedModel,
        global_sparsity: float,
) -> float:
    """
    Compute the magnitude threshold for pruning based on the global sparsity requirement.
    """
    all_weights = []
    for param in model.parameters():
        all_weights.append(
            param.view(-1).abs().clone().detach().cpu()
        )
    all_weights = torch.cat(all_weights)
    # subsample 102400 elements to estimate the threshold
    sample_size = int(min(1e7, all_weights.numel()))
    print(f"[Sparse gradient] Subsampling {sample_size} elements to estimate the threshold.")
    sub_weights = all_weights[torch.randperm(all_weights.numel())[:sample_size]]
    return torch.quantile(sub_weights.float(), global_sparsity).item()


@torch.no_grad()
def compute_named_parameters_to_sparsity(
        model: PreTrainedModel,
        threshold: float,
) -> Dict[str, float]:
    """
    Compute the sparsity of each named parameter in the model.
    """
    named_parameters_to_sparsity = {}
    for name, param in model.named_parameters():
        named_parameters_to_sparsity[name] = param.abs().le(threshold).float().mean().item()
    return named_parameters_to_sparsity
