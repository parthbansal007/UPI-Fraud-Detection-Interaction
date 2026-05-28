from .infer import predict_transaction_risk


def train_transaction_model(config=None):
    from .pipeline import train_transaction_model as _impl

    return _impl(config)


def evaluate_transaction_model(config=None):
    from .pipeline import evaluate_transaction_model as _impl

    return _impl(config)


__all__ = [
    "train_transaction_model",
    "evaluate_transaction_model",
    "predict_transaction_risk",
]
