import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import TimesformerModel


class TimeSformer(nn.Module):
    def __init__(self, num_frames, num_classes=140):
        super(TimeSformer, self).__init__()
        self.num_classes = num_classes
        # BRANCH 1
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                         num_frames=num_frames, ignore_mismatched_sizes=True)

        # ACTION RECOGNITION
        self.group_linear = GroupWiseLinear(self.num_classes, self.backbone.config.hidden_size, bias=True)

    def forward(self, images):
        x = self.backbone(images)[0]
        out = self.group_linear(F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_classes).transpose(1, 2))
        return out


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


