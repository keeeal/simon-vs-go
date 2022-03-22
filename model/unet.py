
from torch import nn

from model.pad import PadConv2DReLU
from model.dense import DenseConv2d, DensePool2d, DenseTran2d


class UNet2d(nn.Module):
    '''A 2D U-Net model.

    Args:
        in_channels (int): The number of channels in the input `x`.
        out_channels (int): The number of channels in the output.
        width (int): The number of filters in the first layer.
        n_conv (int): The number of convolution layers between up/down samples.
        n_pool (int): The number of max-pool downsampling steps.
        kernel_size (int): The size of the convolving kernel.
        pool_size (int): The size of the pooling kernel.

    Inputs: x
        - **x** of shape `(batch, in_channels, width, height)`:
          tensor containing features of the input.
    '''


    def __init__(self, in_channels, out_channels, width=64,
                 n_conv=2, n_pool=4, kernel_size=3, pool_size=2):

        super().__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(*[
                PadConv2DReLU(
                    2**d * width if n else (2**(d-1) * width if d else in_channels),
                    2**d * width, kernel_size
                )
                for n in range(n_conv)
            ])
            for d in range(n_pool)
        ])

        self.pools = nn.ModuleList([
            nn.MaxPool2d(pool_size, ceil_mode=True)
            for d in range(n_pool)
        ])

        self.conv = nn.Sequential(*[
            PadConv2DReLU(
                2**n_pool * width if n else 2**(n_pool - 1) * width,
                2**n_pool * width, kernel_size
            )
            for n in range(n_conv)
        ])

        self.trans = nn.ModuleList([
            DenseTran2d(2**(d+1) * width, 2**d * width, kernel_size,
                          pool_size, output_padding=1)
            for d in range(n_pool)
        ][::-1])

        self.decoders = nn.ModuleList([
            nn.Sequential(*[
                PadConv2DReLU(
                    2**d * width if n else 2**(d+1) * width,
                    2**d * width, kernel_size
                )
                for n in range(n_conv)
            ])
            for d in range(n_pool)
        ][::-1])

        self.out = PadConv2DReLU(width, out_channels, kernel_size)


    def forward(self, x):

        links = []
        for enc, pool in zip(self.encoders, self.pools):
            links.append(enc(x))
            x = pool(links[-1])

        links.reverse()
        x = self.conv(x)
        for link, tran, dec in zip(links, self.trans, self.decoders):
            x = dec(tran(x, link))

        return self.out(x)


class HDUNet2d(nn.Module):
    '''A 2D Hierarchically Dense U-Net model.

    Args:
        in_channels (int): The number of channels in the input `x`.
        out_channels (int): The number of channels in the output.
        growth (int): The number of channels concatenated to `x` at each layer.
        up_factor (int): The number of growth*channels to keep when upsampling.
        n_conv (int): The number of convolution layers between up/down samples.
        n_pool (int): The number of max-pool downsampling steps.
        kernel_size (int): The size of the convolving kernel.
        pool_size (int): The size of the pooling kernel.

    Inputs: x
        - **x** of shape `(batch, in_channels, width, height)`:
          tensor containing features of the input.
    '''


    def __init__(self, in_channels, out_channels, growth, up_factor=4,
                 n_conv=2, n_pool=4, kernel_size=3, pool_size=2):

        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = 2 * [kernel_size]

        padding = [i for k in kernel_size for i in (k // 2, k - k // 2 - 1)]

        i = in_channels
        u = up_factor * growth

        self.encoders = nn.ModuleList([
            nn.Sequential(*[
                DenseConv2d(i + (n + d + d * n_conv) * growth,
                            growth, kernel_size)
                for n in range(n_conv)
            ])
            for d in range(n_pool)
        ])

        self.pools = nn.ModuleList([
            DensePool2d(i + (d + (d + 1) * n_conv) * growth,
                        growth, kernel_size, pool_size)
            for d in range(n_pool)
        ])

        self.conv = nn.Sequential(*[
            DenseConv2d(i + (n + n_pool * (1 + n_conv)) * growth,
                        growth, kernel_size)
            for n in range(2 * n_conv)
        ])

        self.trans = nn.ModuleList([
            DenseTran2d(i + (d + (d + 2) * n_conv) * growth + (d < n_pool) * u,
                        u, kernel_size, pool_size, output_padding=1)
            for d in range(1, n_pool + 1)
        ][::-1])

        self.decoders = nn.ModuleList([
            nn.Sequential(*[
                DenseConv2d(i + (n + d + (d + 1) * n_conv) * growth + u,
                            growth, kernel_size)
                for n in range(n_conv)
            ])
            for d in range(n_pool)
        ][::-1])

        self.out = PadConv2DReLU(
            i + (2 * n_conv) * growth + u,
            out_channels, kernel_size
        )


    def forward(self, x):

        links = []
        for enc, pool in zip(self.encoders, self.pools):
            links.append(enc(x))
            x = pool(links[-1])

        links.reverse()
        x = self.conv(x)
        for link, tran, dec in zip(links, self.trans, self.decoders):
            x = dec(tran(x, link))

        return self.out(x)
