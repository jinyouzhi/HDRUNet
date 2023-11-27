import paddle
import functools
import models.modules.arch_util as arch_util


class HDRUNet(paddle.nn.Layer):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()
        self.conv_first = paddle.nn.Conv2D(in_channels=in_nc, out_channels=
            nf, kernel_size=3, stride=1, padding=1)
        self.SFT_layer1 = arch_util.SFTLayer()
        self.HR_conv1 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.down_conv1 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=2, padding=1)
        self.down_conv2 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=2, padding=1)
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)
        self.up_conv1 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            nf, out_channels=nf * 4, kernel_size=3, stride=1, padding=1),
            paddle.nn.PixelShuffle(upscale_factor=2))
        self.up_conv2 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            nf, out_channels=nf * 4, kernel_size=3, stride=1, padding=1),
            paddle.nn.PixelShuffle(upscale_factor=2))
        self.SFT_layer2 = arch_util.SFTLayer()
        self.HR_conv2 = paddle.nn.Conv2D(in_channels=nf, out_channels=nf,
            kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.conv_last = paddle.nn.Conv2D(in_channels=nf, out_channels=
            out_nc, kernel_size=3, stride=1, padding=1, bias_attr=True)
        cond_in_nc = 3
        cond_nf = 64
        self.cond_first = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels
            =cond_in_nc, out_channels=cond_nf, kernel_size=3, stride=1,
            padding=1), paddle.nn.LeakyReLU(negative_slope=0.1), paddle.nn.
            Conv2D(in_channels=cond_nf, out_channels=cond_nf, kernel_size=1
            ), paddle.nn.LeakyReLU(negative_slope=0.1), paddle.nn.Conv2D(
            in_channels=cond_nf, out_channels=cond_nf, kernel_size=1),
            paddle.nn.LeakyReLU(negative_slope=0.1))
        self.CondNet1 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            cond_nf, out_channels=cond_nf, kernel_size=1), paddle.nn.
            LeakyReLU(negative_slope=0.1), paddle.nn.Conv2D(in_channels=
            cond_nf, out_channels=32, kernel_size=1))
        self.CondNet2 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            cond_nf, out_channels=cond_nf, kernel_size=3, stride=2, padding
            =1), paddle.nn.LeakyReLU(negative_slope=0.1), paddle.nn.Conv2D(
            in_channels=cond_nf, out_channels=32, kernel_size=1))
        self.CondNet3 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            cond_nf, out_channels=cond_nf, kernel_size=3, stride=2, padding
            =1), paddle.nn.LeakyReLU(negative_slope=0.1), paddle.nn.Conv2D(
            in_channels=cond_nf, out_channels=32, kernel_size=3, stride=2,
            padding=1))
        self.mask_est = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            in_nc, out_channels=nf, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(), paddle.nn.Conv2D(in_channels=nf, out_channels
            =nf, kernel_size=3, stride=1, padding=1), paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=nf, out_channels=nf, kernel_size=1
            ), paddle.nn.ReLU(), paddle.nn.Conv2D(in_channels=nf,
            out_channels=out_nc, kernel_size=1))
        if act_type == 'relu':
            self.act = paddle.nn.ReLU()
        elif act_type == 'leakyrelu':
            self.act = paddle.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        mask = self.mask_est(x[0])
        cond = self.cond_first(x[1])
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)
        fea0 = self.act(self.conv_first(x[0]))
        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))
        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))
        fea2 = self.act(self.down_conv2(fea1))
        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2
        out = self.act(self.up_conv1(out)) + fea1
        out, _ = self.recon_trunk3((out, cond2))
        out = self.act(self.up_conv2(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))
        out = self.conv_last(out)
        out = mask * x[0] + out
        return out
