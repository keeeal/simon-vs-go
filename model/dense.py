import torch
from torch import nn

from model.pad import PadConv2DReLU, PadTran2DReLU


def crop(x, shape, center=None):
    center = center if center else len(shape) * [0]
    i = [a // 2 - b // 2 + c for a, b, c in zip(x.shape, shape, center)]
    i = [slice(None) if b < 0 else slice(a, a + b) for a, b in zip(i, shape)]
    return x[tuple(i)]


class DenseConv2d(nn.Module):
    """Applies a 2D convolution layer and nonlinearity then concatenates the
    resulting channels to the original input.

    Args:
        in_channels (int): The number of channels in the input `x`.
        growth (int): The number of new channels concatenated to `x`.
        kernel_size (int): Size of the convolving kernel.

    Inputs: x
        - **x** of shape `(batch, in_channels, depth, width, height)`:
          tensor containing features of the input.
    """

    def __init__(self, in_channels, growth, kernel_size):
        super().__init__()
        self.conv = PadConv2DReLU(in_channels, growth, kernel_size)

    def forward(self, x):
        return torch.cat((x, self.conv(x)), dim=1)


class DensePool2d(nn.Module):
    """Applies a 2D convolution layer and nonlinearity then concatenates the
    resulting channels to the downsampled original input.

    Args:
        in_channels (int): The number of channels in the input `x`.
        growth (int): The number of new channels concatenated to `x`.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Size of the convolving stride and pooling kernel.

    Inputs: x
        - **x** of shape `(batch, in_channels, depth, width, height)`:
          tensor containing features of the input.
    """

    def __init__(self, in_channels, growth, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool2d(stride, ceil_mode=True)
        self.conv = PadConv2DReLU(in_channels, growth, kernel_size, stride)

    def forward(self, x):
        return torch.cat((self.pool(x), self.conv(x)), dim=1)


class DenseTran2d(nn.Module):
    """Applies a transposed 2D convolution layer and nonlinearity then
    concatenates the resulting channels to another tensor.

    Args:
        in_channels (int): The number of channels in the input `x`.
        growth (int): The number of new channels concatenated to `x`.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Size of the convolving stride and pooling kernel.
        output_padding (int): The additional size added to one side of the
            output shape.
        dropout (float): The probability of dropout applied to input `x`.

    Inputs: x
        - **x** of shape `(batch, in_channels, depth, width, height)`:
          tensor containing features of the input.
        - **x_cat** of shape `(batch, channels, depth, width, height)`:
          tensor containing features to be concatenated.
    """

    def __init__(self, in_channels, growth, kernel_size, stride, output_padding=0):

        super().__init__()
        self.tran = PadTran2DReLU(
            in_channels, growth, kernel_size, stride, output_padding
        )

    def forward(self, x, x_cat):
        shape = 2 * [-1] + list(x_cat.shape)[2:]
        return torch.cat((x_cat, crop(self.tran(x), shape)), dim=1)
