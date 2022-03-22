from torch import nn


def get_padding(kernel_size, dims):
    if isinstance(kernel_size, int):
        kernel_size = dims * [kernel_size]

    return [i for k in kernel_size for i in (k // 2, k - k // 2 - 1)]


class PadConv2DReLU(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

        super().__init__()
        self.conv = nn.Sequential(
            nn.ConstantPad2d(get_padding(kernel_size, 2), 0),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PadTran2DReLU(nn.Module):
    """
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, output_padding=0
    ):

        super().__init__()
        self.tran = nn.Sequential(
            nn.ConstantPad2d(get_padding(kernel_size, 2), 0),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                output_padding=output_padding,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.tran(x)
