🌄 Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark
====================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[[Paper]](https://arxiv.org/pdf/2402.11592.pdf) [[Code]](https://github.com/ZO-Bench/ZO-LLM) [[Website]](https://sites.google.com/view/zo-tutorial-aaai-2024/)

Official code for the paper **"Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark
"**.

Authors (* Equal Contribution): _[Yihua Zhang](https://www.yihua-zhang.com/)\*, [Pingzhi Li](https://pingzhili.github.io/)\*,
[Junyuan Hong](https://jyhong.gitlab.io/)\*, [Jiaxiang Li](https://jasonjiaxiangli.github.io/)\*,
[Yimeng Zhang](https://damon-demon.github.io/), [Wenqing Zheng](https://wenqing-zheng.github.io/),
[Pin-Yu Chen](https://sites.google.com/site/pinyuchenpage/home), [Jason D. Lee](https://jasondlee88.github.io/),
[Wotao Yin](https://wotaoyin.mathopt.com/), [Mingyi Hong](https://people.ece.umn.edu/~mhong/mingyi.html),
[Zhangyang Wang](https://vita-group.github.io/group.html), [Sijia Liu](https://lsjxjtu.github.io/),
and [Tianlong Chen](https://tianlong-chen.github.io/)_

### Table of Contents

1. **[Overview](#1-overview)**
2. **[Project Structure](#2-project-structure)**
3. **[Getting Started](#3-getting-started)**
4. **[Reproducing Results](#4-reproducing-results)**
5. **[Citation](#5-citation)**

## 1) Overview

This repo contains the source code and reproducing guide of ZO-LLM.
This research endeavor is designed to help researchers better understand the capabilities, limitations and principles
associated with the BP-free, zeroth-order (ZO) optimization as a solution for reducing memory costs during Large
Language Model (LLM) fine-tuning. Our study unveils previously overlooked optimization principles,
highlighting the importance of task alignment, the role of the forward gradient method,
and the balance between algorithm complexity and fine-tuning performance.

This project is organized around the following scopes, including:

1. **Five** LLM families: Roberta, OPT, LLaMA, Vicuna, and Mistral.
2. **Three** task complexities: binary classification, question-answering, and commonsense reasoning.
3. **Four** fine-tuning schemes: full fine-tuning, LoRA, prefix tuning, and prompt tuning.
4. **Six** BP-free optimization methods: ZO-SGD, ZO-SGD-Sign, ZO-SGD-MMT, ZO-SGD-Cons, ZO-Adam, and forward gradient.
5. **Three** novel enhancements to ZO optimization: block-wise descent, hybrid training, and gradient sparsity.

## 2) Project Structure

This project is structured around the hyperparameter sweeping for various tasks & models & tuning schemes & optimization
methods. All optimization methods are implemented in `zo-bench/trainer.py`. Task configurations are defined in
`zo-bench/tasks.py` and `zo-bench/templates.py`. The main entry point is `zo-bench/run.py`.

```
.
├── zo-bench
│   ├── modeling_mistral
│   │   ├─── __init__.py
│   │   ├── configuration_mistral.py
│   │   ├── modleing_mistral.py
│   ├── modeling_llama.py
│   ├── modeling_opt.py
│   ├── modeling_roberta.py
│   ├── prefix_tuning.py
│   ├── prompt_tuning.py
│   ├── run.py
│   ├── tasks.py
│   ├── templates.py
│   ├── test_fake_text_memory.py
│   ├── trainer.py
│   ├── utils.py
│   ├── sweep
│   │   ├── Copa_llama-7b
│   │   │   ├── adam
│   │   │   │   ├── adam_copa_ft.yml
│   │   │   │   ├── adam_copa_lora.yml
│   │   │   │   ├── adam_copa_prefix.yml
│   │   │   │   ├── adam_copa_prompt.yml
│   │   │   ├── forward_grad
│   │   │   │   ├── forward_grad_copa_ft.yml
│   │   │   │   ├── forward_grad_copa_lora.yml
│   │   │   │   ├── forward_grad_copa_prefix.yml
│   │   │   │   ├── forward_grad_copa_prompt.yml
│   │   │   ├── sgd
│   │   │   │   ├── sgd_copa_ft.yml
│   │   │   │   ├── sgd_copa_lora.yml
│   │   │   │   ├── sgd_copa_prefix.yml
│   │   │   │   ├── sgd_copa_prompt.yml
│   │   │   ├── sign_sgd
│   │   │   │   ├── sign_sgd_copa_ft.yml
│   │   │   │   ├── sign_sgd_copa_lora.yml
│   │   │   │   ├── sign_sgd_copa_prefix.yml
│   │   │   │   ├── sign_sgd_copa_prompt.yml
│   │   │   ├── zo_adam
│   │   │   │   ├── zo_adam_copa_ft.yml
│   │   │   │   ├── zo_adam_copa_lora.yml
│   │   │   │   ├── zo_adam_copa_prefix.yml
│   │   │   │   ├── zo_adam_copa_prompt.yml
│   │   │   ├── zo_sgd
│   │   │   │   ├── zo_sgd_copa_ft.yml
│   │   │   │   ├── zo_sgd_copa_lora.yml
│   │   │   │   ├── zo_sgd_copa_prefix.yml
│   │   │   │   ├── zo_sgd_copa_prompt.yml
│   │   │   ├── zo_sgd_conserv
│   │   │   │   ├── zo_sgd_conserv_copa_ft.yml
│   │   │   │   ├── zo_sgd_conserv_copa_lora.yml
│   │   │   │   ├── zo_sgd_conserv_copa_prefix.yml
│   │   │   │   ├── zo_sgd_conserv_copa_prompt.yml
│   │   │   ├── zo_sgd_momen
│   │   │   │   ├── zo_sgd_momen_copa_ft.yml
│   │   │   │   ├── zo_sgd_momen_copa_lora.yml
│   │   │   │   ├── zo_sgd_momen_copa_prefix.yml
│   │   │   │   ├── zo_sgd_momen_copa_prompt.yml
│   │   ├── Copa_llama-13b
│   │   │   ├── ...
│   │   ├── Copa_mistral
│   │   │   ├── ...
│   │   ├── Copa_opt-13b
│   │   │   ├── ...
│   │   ├── Copa_vicuna
│   │   │   ├── ...
│   │   ├── SST2_opt-1.3b
│   │   │   ├── ...
│   │   ├── WinoGrande_llama-7b
│   │   │   ├── ...
│   │   ├── WinoGrande_llama-13b
│   │   │   ├── ...
│   │   ├── WinoGrande_mistral
│   │   │   ├── ...
│   │   ├── WinoGrande_opt-13b
│   │   │   ├── ...
│   │   ├── WinoGrande_vicuna
│   │   │   ├── ...
├── environment.yml
```

## 3) Getting Started

All you need is:

```bash
conda create -n zollm python=3.10
conda activate zollm
pip install -r requirements.txt
```

## 4) Reproducing Results

We provide detailed hyperparameter settings in [sweeps](zo-bench/sweeps), 
where the sweep configuration for tuning a MODEL on TASK under SCHEME with OPTIMIZER is organized as `zo-bench/sweeps/TASK_MODEL/OPTIMIZER/SCHEME.yml`.

An example use of sweep for full fine-tuning LLaMA-7B with ZO-SGD on the COPA task is as follows:

```
~> wandb sweep zo-bench/sweeps/Copa_llama-7b/zo_sgd/zo_sgd_copa_ft.yml
wandb: Creating sweep from: zo-bench/sweeps/Copa_llama-7b/zo_sgd/zo_sgd_copa_ft.yml
wandb: Created sweep with ID: <ID>
wandb: View sweep at: https://wandb.ai/<unique ID>
wandb: Run sweep agent with: wandb agent <unique ID>
~> wandb agent <unique ID>
```

## 5) Citation

```
@misc{zhang2024revisiting,
      title={Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark}, 
      author={Yihua Zhang and Pingzhi Li and Junyuan Hong and Jiaxiang Li and Yimeng Zhang and Wenqing Zheng and Pin-Yu Chen and Jason D. Lee and Wotao Yin and Mingyi Hong and Zhangyang Wang and Sijia Liu and Tianlong Chen},
      year={2024},
      eprint={2402.11592},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
