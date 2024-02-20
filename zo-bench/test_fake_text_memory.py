# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/8
import numpy as np
import torch
from fire import Fire
from torch.optim import SGD
from transformers import AutoModelForCausalLM

zo_eps = 1e-3


def zo_perturb_parameters(named_parameters_to_optim, random_seed=None, scaling_factor=1):
    """
    Perturb the parameters with random vector z.
    Input:
    - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
    - scaling_factor: theta = theta + scaling_factor * z * eps
    """

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(random_seed if random_seed is not None else 0)

    for _, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * zo_eps


def zo_forward(model, inputs):
    """
    Get (no gradient) loss from the model. Dropout is turned off too.
    """
    model.eval()
    with torch.inference_mode():
        input_ids = inputs["input_ids"]

        # Prepare labels for loss calculation (shifted input ids)
        labels = input_ids.clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = -100  # We don't need to compute loss for the last token

        # Step 5: Perform a forward pass and compute loss
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
    return loss.detach()


@torch.no_grad()
def zo_step(model, inputs, optimizer, is_sign_opt=False):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_parameters_to_optim.append((name, param))
            # # TODO avoid init the memory for grad.
            # param.grad = torch.zeros_like(param.data)
            param.grad = None  # Make sure the grad is empty and will not be updated.

    # Sample the random seed for sampling z
    zo_random_seed = np.random.randint(1000000000)

    # First function evaluation
    zo_perturb_parameters(named_parameters_to_optim, scaling_factor=1)
    loss1 = zo_forward(model, inputs)

    # Second function evaluation
    zo_perturb_parameters(named_parameters_to_optim, scaling_factor=-2)
    loss2 = zo_forward(model, inputs)
    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

    # Reset model back to its parameters at start of step
    zo_perturb_parameters(named_parameters_to_optim, scaling_factor=1)

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(zo_random_seed)
    for name, param in named_parameters_to_optim:
        # Resample z
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                         dtype=param.data.dtype)

        if is_sign_opt:
            # ----signOpt_orig
            # TODo why do we multiply lr here? We will multiply lr twice?
            graddiff_times_z = np.sign(projected_grad) * z
            # ----signOpt_mul_sign
            # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
        else:
            # ----mezo original
            graddiff_times_z = projected_grad * z

        # param.grad += graddiff_times_z.detach()
        # more mem-efficient:
        # run optimizer.step here to avoid caching all grad.
        param.grad = graddiff_times_z
        optimizer.step()  # will only update grad that is not None.
        # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        param.grad = None  # avoid further update.

    return loss1


def fo_step(model, inputs, optimizer):
    input_ids = inputs["input_ids"]

    # Prepare labels for loss calculation (shifted input ids)
    labels = input_ids.clone()
    labels[:, :-1] = labels[:, 1:].clone()
    labels[:, -1] = -100  # We don't need to compute loss for the last token

    # Perform a forward pass and compute loss
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    # first order step
    loss.backward()
    optimizer.step()
    model.no_grad()
    return loss.detach()


from functools import partial
from torch.func import functional_call, jvp

lr = 1e-6


# @staticmethod
@torch.no_grad()
def functional_call_loss(params, names, buffers, model, batch):
    params = {k: v for k, v in zip(names, params)}
    outputs = functional_call(model, (params, buffers), tuple(), kwargs=batch)
    return outputs[0]  # only need the loss
    # return outputs


def forward_gradient_step(model, inputs, optimizer):
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_parameters_to_optim.append((name, param))
            param.grad = None
            # this is not memory efficient.
            # param.grad = torch.zeros_like(param.data)

    # Sample the random seed for sampling vs
    zo_random_seed = np.random.randint(1000000000)
    torch.manual_seed(zo_random_seed)

    cur_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    loss = 0
    print(f"Init MEM: {cur_mem:.2f} GB")
    vs = [torch.randn_like(p) for _, p in named_parameters_to_optim]

    # fixme: this is a workaround for device map error when using jvp
    inputs = {
        k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    f = partial(
        functional_call_loss,
        names=[n for n, _ in named_parameters_to_optim], buffers=dict(model.named_buffers()),
        model=model, batch=inputs
    )
    print(f"MEM increase for vs: {torch.cuda.max_memory_allocated() / 1024 ** 3 - cur_mem:.2f} GB")
    cur_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    # jvp profiling
    loss_, jvp_ = jvp(f, (list([p for _, p in named_parameters_to_optim]),), (vs,))

    print(f"MEM increase for jvp: {torch.cuda.max_memory_allocated() / 1024 ** 3 - cur_mem:.2f} GB")

    if isinstance(jvp_, tuple):
        jvp_ = jvp_[0]
        print("WARNING: jvp return useless tensors, which may consume a lot of memory.")
    with torch.no_grad():
        for v, (n, p) in zip(vs, named_parameters_to_optim):
            p.data.sub_(lr * (v * jvp_.to(p.device)))
    loss += loss_.item()

    return torch.tensor(loss)


def main_test(
        batch_size: int,
        sequence_length: int,
        use_amp: bool = False,
        load_float16: bool = True,
):
    model_name = "facebook/opt-13b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if load_float16 else torch.float32,
    )

    max_memory_allocated = 0
    for device_id in range(torch.cuda.device_count()):
        max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
    print(f"[bsz={batch_size} | seq_len={sequence_length}]"
          f"init peak mem (model weight): {max_memory_allocated / 1024 ** 3:.2f} GB")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = SGD(model.parameters(), lr=1e-5)  # , momentum=0.9)

    # Step 4: Preprocess the input
    # text = "Replace this with your text."
    inputs = {
        "input_ids": torch.randint(5000, 10000, (batch_size, sequence_length), dtype=torch.long).cuda(),
        "attention_mask": torch.zeros(batch_size, sequence_length, dtype=torch.long).cuda()
    }
    for _ in range(1):
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(device_id)
        # loss = zo_step(model, inputs, optimizer)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            labels[:, :-1] = labels[:, 1:].clone()
            labels[:, -1] = -100  # We don't need to compute loss for the last token
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        max_memory_allocated = 0
        for device_id in range(torch.cuda.device_count()):
            max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
        print(f"[bsz={batch_size} | seq_len={sequence_length}]"
              f"step peak mem (during fwd): {max_memory_allocated / 1024 ** 3:.2f} GB")
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(device_id)

        scaler.scale(loss).backward()

        max_memory_allocated = 0
        for device_id in range(torch.cuda.device_count()):
            max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
        print(f"[bsz={batch_size} | seq_len={sequence_length}]"
              f"step peak mem (during bwd): {max_memory_allocated / 1024 ** 3:.2f} GB")
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(device_id)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        max_memory_allocated = 0
        for device_id in range(torch.cuda.device_count()):
            max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
        print(f"[bsz={batch_size} | seq_len={sequence_length}]"
              f"step peak mem (during optim): {max_memory_allocated / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    Fire(main_test)
