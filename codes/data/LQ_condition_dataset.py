import paddle
import numpy as np
import data.util as util


class LQ_Dataset(paddle.io.Dataset):
    """Read LQ images only in the test phase."""

    def __init__(self, opt):
        super(LQ_Dataset, self).__init__()
        self.opt = opt
        self.paths_LQ = None
        self.LQ_env = None
        print(opt['data_type'], opt['dataroot_LQ'])
        self.LQ_env, self.paths_LQ = util.get_image_paths(opt['data_type'],
            opt['dataroot_LQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'

    def __getitem__(self, index):
        LQ_path = None
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_img(self.LQ_env, LQ_path)
        H, W, C = img_LQ.shape
        if self.opt['condition'] == 'image':
            cond = img_LQ.copy()
        elif self.opt['condition'] == 'gradient':
            cond = util.calculate_gradient(img_LQ)
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            cond = cond[:, :, [2, 1, 0]]
        img_LQ = paddle.to_tensor(data=np.ascontiguousarray(np.transpose(
            img_LQ, (2, 0, 1)))).astype(dtype='float32')
        cond = paddle.to_tensor(data=np.ascontiguousarray(np.transpose(cond,
            (2, 0, 1)))).astype(dtype='float32')
        return {'LQ': img_LQ, 'LQ_path': LQ_path, 'cond': cond}

    def __len__(self):
        return len(self.paths_LQ)
