import torch
from transformers import get_scheduler

from speedrunning_plms.optim import Muon
from speedrunning_plms.training.utils import LerpFloat, LerpTensor


def build_optimizers(model, args, print_fn=print):
    if args.use_muon:
        matrix_params = [
            p for n, p in model.named_parameters()
            if p.ndim >= 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
        ]
        embed_params = [
            p for n, p in model.named_parameters() if "embed" in n.lower() and p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters() if "lm_head" in n.lower() and p.requires_grad
        ]
        scalar_params = [
            p for n, p in model.named_parameters()
            if p.ndim < 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
        ]

        all_params = [p for p in model.parameters() if p.requires_grad]
        mapped_params = matrix_params + embed_params + head_params + scalar_params
        assert len(all_params) == len(mapped_params), (
            f"Muon parameter mapping mismatch: {len(all_params)} total vs {len(mapped_params)} mapped"
        )
        print_fn(
            f"Muon optimizer initialized: {len(matrix_params)} matrix, {len(embed_params)} embed, "
            f"{len(head_params)} head, {len(scalar_params)} scalar params. Total: {len(all_params)}"
        )

        optimizer1 = torch.optim.Adam([
            dict(params=embed_params, lr=args.lr_embed),
            dict(params=head_params, lr=args.lr_head),
            dict(params=scalar_params, lr=args.lr_scalar),
        ], betas=(0.8, 0.95), fused=True)
        optimizer2 = Muon(matrix_params, lr=args.lr_hidden, momentum=0.95)
        return [optimizer1, optimizer2]

    params = [p for p in model.parameters() if p.requires_grad]
    print_fn(f"AdamW optimizer initialized with {len(params)} parameters.")
    return [torch.optim.AdamW(params, lr=args.lr)]


def build_schedulers(optimizers, args):
    lr_schedulers = []
    adam_scheduler = get_scheduler(
        args.scheduler_type,
        optimizer=optimizers[0],
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_steps,
    )
    lr_schedulers.append(adam_scheduler)
    if args.use_muon:
        muon_scheduler = get_scheduler(
            args.scheduler_type,
            optimizer=optimizers[-1],
            num_warmup_steps=0,
            num_training_steps=args.num_steps,
        )
        lr_schedulers.append(muon_scheduler)

    sliding_window_size_scheduler = LerpTensor(start_val=1024, end_val=args.max_length, precision=128)
    if args.mask_rate_schedule:
        mask_rate_scheduler = LerpFloat(
            start_val=args.starting_mask_rate,
            end_val=args.mask_rate,
            precision=0.01,
        )
    else:
        mask_rate_scheduler = None
    return lr_schedulers, sliding_window_size_scheduler, mask_rate_scheduler


def apply_muon_momentum_warmup(optimizer, step: int, warmup_steps: int) -> None:
    frac = min(step / warmup_steps, 1)
    for group in optimizer.param_groups:
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
