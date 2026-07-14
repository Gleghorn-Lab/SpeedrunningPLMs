from argparse import Namespace

from speedrunning_plms.models import PLMConfig


def apply_bugfix_overrides(args: Namespace) -> None:
    if not args.bugfix:
        return
    args.hidden_size = 128
    args.num_attention_heads = 2
    args.num_hidden_layers = 2
    args.expansion_ratio = 2.0
    args.soft_logit_cap = 16.0
    args.tie_embeddings = False
    args.unet = True
    args.batch_size = 2048
    args.grad_accum = 1
    args.num_steps = 10
    args.cooldown_steps = 2
    args.max_length = 512
    args.auto_grad_clip = True
    args.grad_clip = 0.0


def validate_args(args: Namespace) -> None:
    if args.mlm and args.masked_diffusion:
        raise ValueError("Only one of --mlm or --masked_diffusion can be true.")
    if args.auto_grad_clip and args.grad_clip > 0:
        raise ValueError("Cannot use both --auto_grad_clip and --grad_clip at the same time. Choose one.")
    if getattr(args, "push_to_hub", False) and not getattr(args, "hf_model_name", None):
        raise ValueError("--hf_model_name is required when --push_to_hub is enabled.")


def build_model_config(args: Namespace) -> PLMConfig:
    return PLMConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        num_unet_layers=args.num_unet_layers,
        num_extra_layers=args.num_extra_layers,
        max_sequence_length=args.max_length,
        vocab_size=args.vocab_size,
        expansion_ratio=args.expansion_ratio,
        soft_logit_cap=args.soft_logit_cap,
        tie_embeddings=args.tie_embeddings,
        unet=args.unet,
        patch_unet=args.patch_unet,
        mlm=args.mlm or args.masked_diffusion,
        masked_diffusion=args.masked_diffusion,
        token_dropout=args.token_dropout,
        compile_flex_attention=args.compile_flex_attention,
    )
