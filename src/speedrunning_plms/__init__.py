__all__ = ["PLM", "PLMConfig"]


def __getattr__(name: str):
    if name in {"PLM", "PLMConfig"}:
        from speedrunning_plms.models import PLM, PLMConfig

        return {"PLM": PLM, "PLMConfig": PLMConfig}[name]
    raise AttributeError(name)
