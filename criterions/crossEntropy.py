from logging import Logger
from typing import Any, Dict
from torch.nn import Conv2d, CrossEntropyLoss, Linear, Module, Sequential


class CrossEntropy(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(CrossEntropy, self).__init__()
        self.name = "CrossEntropyLoss"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.criterion = CrossEntropyLoss()
        self.log.debug("Initialised CrossEntropy criterion.")
        self.fc = Linear(512, config["num_classes"])
        self.to(config["device"])

    def forward(self, features, labels):
        """
        Need argmax of labels
        """
        preds = self.fc(features)
        return self.criterion(preds, labels.argmax(dim=1)), preds
