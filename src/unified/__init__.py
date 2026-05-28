def train_unified_fusion(config=None):
    from .fusion import train_unified_fusion as _impl

    return _impl(config)


def infer_unified_risk(df_or_inputs, config=None):
    from .fusion import infer_unified_risk as _impl

    return _impl(df_or_inputs, config=config)


def save_fusion_config(config, path=None):
    from .fusion import save_fusion_config as _impl

    if path is None:
        return _impl(config)
    return _impl(config, path=path)


def evaluate_unified_model(config=None):
    from .evaluate import evaluate_unified_model as _impl

    return _impl(config)


__all__ = [
    "train_unified_fusion",
    "infer_unified_risk",
    "evaluate_unified_model",
    "save_fusion_config",
]
