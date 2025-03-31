from logging import Logger
from typing import Any, Dict 
from torch.nn import Conv2d, Module, Sequential


class SimpleAttention(Module):

    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(SimpleAttention, self).__init__()
        self.name = "simpleAttention"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised simpleAttention model.")

    def forward(self, x):
        raise NotImplementedError()
