import paddle


class tanh_L1Loss(paddle.nn.Layer):

    def __init__(self):
        super(tanh_L1Loss, self).__init__()

    def forward(self, x, y):
        loss = paddle.mean(x=paddle.abs(x=paddle.nn.functional.tanh(x=x) -
            paddle.nn.functional.tanh(x=y)))
        return loss


class tanh_L2Loss(paddle.nn.Layer):

    def __init__(self):
        super(tanh_L2Loss, self).__init__()

    def forward(self, x, y):
        loss = paddle.mean(x=paddle.pow(x=paddle.nn.functional.tanh(x=x) -
            paddle.nn.functional.tanh(x=y), y=2))
        return loss
