from speedrunning_plms.training.runtime import *  # noqa: F401,F403
from speedrunning_plms.training.config import (
    apply_bugfix_overrides,
    build_model_config,
    validate_args,
)
from speedrunning_plms.training.trainer import Trainer, arg_parser, build_code_snapshot, set_code_snapshot


def main() -> None:
    args = arg_parser()
    apply_bugfix_overrides(args)
    validate_args(args)
    model_config = build_model_config(args)

    wandb_initialized = False
    if args.wandb_token:
        import os

        if os.environ.get("WANDB_AVAILABLE") == "true":
            import wandb

            wandb.login(key=args.wandb_token)
            wandb_initialized = True

    if args.hf_token:
        from huggingface_hub import login

        login(args.hf_token)
        args.hf_token = None

    if args.wandb_token:
        args.wandb_token = None

    set_code_snapshot(build_code_snapshot())
    trainer = Trainer(args, model_config)
    trainer.wandb_initialized = wandb_initialized
    trainer.train()


if __name__ == "__main__":
    main()
