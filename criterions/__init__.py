from logging import Logger
from typing import Any, Dict
from torch.nn import Module


def get_criterion(model: str, config: Dict[str, Any], log: Logger, **kwargs) -> Module:


    if model == "crossEntropy":
        from criterions.crossEntropy import CrossEntropy

        return CrossEntropy(config, log, **kwargs)


    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Model: {model} not present")
