from logging import Logger
from typing import Any, Dict
from torch.nn import (
    Conv2d,
    Linear,
    Module,
    MultiheadAttention,
    ReLU,
    BatchNorm2d,
)
import torch.nn.functional as F
import torch


class SimpleAttention(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(SimpleAttention, self).__init__()
        self.name = "simpleAttention"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised simpleAttention model.")
        # input is (B,101,75,40)
        self.conv1 = Conv2d(101, 128, kernel_size=3, stride=2)
        self.norm1 = BatchNorm2d(128)
        self.conv2 = Conv2d(128, 256, kernel_size=3, stride=2)
        self.norm2 = BatchNorm2d(256)
        self.relu = ReLU()
        self.conv3 = Conv2d(256, 512, kernel_size=3, stride=2)
        self.norm3 = BatchNorm2d(512)
        self.keytransform = Linear(512, 512)
        self.valuetransform = Linear(512, 512)
        self.querytransform = Linear(512, 512)
        self.attention = MultiheadAttention(
            embed_dim=512, num_heads=4, batch_first=True
        )
        self.embed_transform = Linear(512, 512)
        self.fuser = Linear(2, 1)
        self.to(config["device"])

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x.view(x.size(0), x.size(1), -1)
        x_before_attention = x.permute(0, 2, 1)
        x_after_attention, _ = self.attention(
            self.querytransform(x_before_attention),
            self.keytransform(x_before_attention),
            self.valuetransform(x_before_attention),
        )
        x = x_after_attention + x_before_attention
        x = self.embed_transform(x)
        x = x.permute(0, 2, 1)
        avgpool = F.adaptive_avg_pool1d(x, 1)
        maxpool = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([avgpool, maxpool], dim=2)
        x = self.fuser(x)
        x = x.squeeze(2)

        return x
