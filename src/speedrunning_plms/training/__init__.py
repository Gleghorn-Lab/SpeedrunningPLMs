__all__ = [
    "Trainer",
    "apply_bugfix_overrides",
    "arg_parser",
    "build_model_config",
    "validate_args",
]


def __getattr__(name: str):
    if name in {"apply_bugfix_overrides", "build_model_config", "validate_args"}:
        from speedrunning_plms.training.config import (
            apply_bugfix_overrides,
            build_model_config,
            validate_args,
        )

        return {
            "apply_bugfix_overrides": apply_bugfix_overrides,
            "build_model_config": build_model_config,
            "validate_args": validate_args,
        }[name]
    if name in {"Trainer", "arg_parser"}:
        from speedrunning_plms.training.trainer import Trainer, arg_parser

        return {"Trainer": Trainer, "arg_parser": arg_parser}[name]
    raise AttributeError(name)
