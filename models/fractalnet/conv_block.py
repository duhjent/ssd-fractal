import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Conv - Dropout - BN - ReLU"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout=None,
        pad_type="zero",
        dropout_pos="CDBR",
    ):
        """Conv
        Args:
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal-dropout
        """
        super().__init__()
        self.dropout_pos = dropout_pos
        if pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == "reflect":
            # [!] the paper used reflect padding - just for data augmentation?
            self.pad = nn.ReflectionPad2d(padding)
        else:
            raise ValueError(pad_type)

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=0, bias=False)
        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = None
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.dropout_pos == "CDBR" and self.dropout:
            out = self.dropout(out)
        out = self.bn(out)
        out = F.relu_(out)
        if self.dropout_pos == "CBRD" and self.dropout:
            out = self.dropout(out)

        return out
