import paddle


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
                    negative_slope=0, nonlinearity='leaky_relu')
                init_KaimingNormal(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, paddle.nn.Linear):
                init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
                    negative_slope=0, nonlinearity='leaky_relu')
                init_KaimingNormal(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, paddle.nn.BatchNorm2D):
                init_Constant = paddle.nn.initializer.Constant(value=1)
                init_Constant(m.weight)
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(m.bias.data)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return paddle.nn.Sequential(*layers)


class ResidualBlock_noBN(paddle.nn.Layer):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.conv2 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1, bias_attr=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = paddle.nn.functional.relu(x=self.conv1(x))
        out = self.conv2(out)
        return identity + out


class SFTLayer(paddle.nn.Layer):

    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = paddle.nn.Conv2D(in_channels=in_nc,
            out_channels=nf, kernel_size=1)
        self.SFT_scale_conv1 = paddle.nn.Conv2D(in_channels=nf,
            out_channels=out_nc, kernel_size=1)
        self.SFT_shift_conv0 = paddle.nn.Conv2D(in_channels=in_nc,
            out_channels=nf, kernel_size=1)
        self.SFT_shift_conv1 = paddle.nn.Conv2D(in_channels=nf,
            out_channels=out_nc, kernel_size=1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(paddle.nn.functional.leaky_relu(x=self
            .SFT_scale_conv0(x[1]), negative_slope=0.1))
        shift = self.SFT_shift_conv1(paddle.nn.functional.leaky_relu(x=self
            .SFT_shift_conv0(x[1]), negative_slope=0.1))
        return x[0] * (scale + 1) + shift


class ResBlock_with_SFT(paddle.nn.Layer):

    def __init__(self, nf=64):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.conv2 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.sft1 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv1 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1)
        self.sft2 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv2 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        fea = self.sft1(x)
        fea = paddle.nn.functional.relu(x=self.conv1(fea))
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return x[0] + fea, x[1]


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.shape[-2:] == flow.shape[1:3]
    B, C, H, W = x.shape
    grid_y, grid_x = paddle.meshgrid(paddle.arange(start=0, end=H), paddle.
        arange(start=0, end=W))
    paddle.device.set_device("intel_gpu")
    grid = paddle.stack(x=(grid_x, grid_y), axis=2).astype(dtype='float32')
    grid.stop_gradient = not False
    grid = grid.astype(dtype=x.dtype)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = paddle.stack(x=(vgrid_x, vgrid_y), axis=3)
    output = paddle.nn.functional.grid_sample(x=x, grid=vgrid_scaled, mode=
        interp_mode, padding_mode=padding_mode)
    return output
