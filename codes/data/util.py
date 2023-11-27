import sys
sys.path.append('/home/youzhiji/workload/paddle_project/utils')
import paddle_aux
import paddle
import os
import math
import pickle
import random
import numpy as np
import cv2
import scipy.ndimage
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
    '.PPM', '.bmp', '.BMP', '.tif', '.npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb')
        )
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'
                .format(data_type))
    return sizes, paths


def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:
        if os.path.splitext(path)[1] == '.npy':
            img = np.load(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_npy(path):
    return np.load(path)


def read_imgdata(path, ratio=255.0):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) / ratio


def expo_correct(img, exposures, idx):
    floating_exposures = exposures - exposures[1]
    gamma = 2.24
    img_corrected = (img ** gamma * 2.0 ** (-1 * floating_exposures[idx])) ** (
        1 / gamma)
    return img_corrected


def augment(img_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow
    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]
    return rlt_img_list, rlt_flow_list


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def calculate_gradient(img, ksize=-1):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobelxy.astype(np.float32) / 255.0


def cubic(x):
    absx = paddle.abs(x=x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1).astype(dtype=absx.
        dtype) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((absx > 1) *
        (absx <= 2)).astype(dtype=absx.dtype)


def calculate_weights_indices(in_length, out_length, scale, kernel,
    kernel_width, antialiasing):
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale
    x = paddle.linspace(start=1, stop=out_length, num=out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = paddle.floor(x=u - kernel_width / 2)
    P = math.ceil(kernel_width) + 2
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    indices = left.reshape([out_length, 1]).expand(shape=[out_length, P]
        ) + paddle.linspace(start=0, stop=P - 1, num=P).reshape([1, P]).expand(shape
        =[out_length, P])
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    distance_to_center = u.reshape([out_length, 1]).expand(shape=[out_length, P]
        ) - indices
    if scale < 1 and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    weights_sum = paddle.sum(x=weights, axis=1).reshape([out_length, 1])
    weights = weights / weights_sum.expand(shape=[out_length, P])
    weights_zero_tmp = paddle.sum(x=weights == 0, axis=0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-06):
        start_0 = indices.shape[1] + 1 if 1 < 0 else 1
        indices = paddle.slice(indices, [1], [start_0], [start_0 + (P - 2)])
        start_1 = weights.shape[1] + 1 if 1 < 0 else 1
        weights = paddle.slice(weights, [1], [start_1], [start_1 + (P - 2)])
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-06):
        start_2 = indices.shape[1] + 0 if 0 < 0 else 0
        indices = paddle.slice(indices, [1], [start_2], [start_2 + (P - 2)])
        start_3 = weights.shape[1] + 0 if 0 < 0 else 0
        weights = paddle.slice(weights, [1], [start_3], [start_3 + (P - 2)])
    weights = weights
    indices = indices
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    in_C, in_H, in_W = img.shape
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    img_aug = paddle.empty(shape=[in_C, in_H + sym_len_Hs + sym_len_He,
        in_W], dtype='float32')
    start_4 = img_aug.shape[1] + sym_len_Hs if sym_len_Hs < 0 else sym_len_Hs
    paddle.assign(img, output=paddle.slice(img_aug, [1], [start_4], [
        start_4 + in_H]))
    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = paddle.arange(start=sym_patch.shape[1] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=1, index=inv_idx)
    start_5 = img_aug.shape[1] + 0 if 0 < 0 else 0
    paddle.assign(sym_patch_inv, output=paddle.slice(img_aug, [1], [start_5
        ], [start_5 + sym_len_Hs]))
    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = paddle.arange(start=sym_patch.shape[1] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=1, index=inv_idx)
    start_6 = img_aug.shape[1
        ] + sym_len_Hs + in_H if sym_len_Hs + in_H < 0 else sym_len_Hs + in_H
    paddle.assign(sym_patch_inv, output=paddle.slice(img_aug, [1], [start_6
        ], [start_6 + sym_len_He]))
    out_1 = paddle.empty(shape=[in_C, out_H, in_W], dtype='float32')
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        x = img_aug[0, idx:idx + kernel_width, :]
        perm_0 = list(range(x.ndim))
        perm_0[0] = 1
        perm_0[1] = 0
        out_1[0, i, :] = x.transpose(perm=perm_0).mv(vec=weights_H[i])
        x = img_aug[1, idx:idx + kernel_width, :]
        perm_1 = list(range(x.ndim))
        perm_1[0] = 1
        perm_1[1] = 0
        out_1[1, i, :] = x.transpose(perm=perm_1).mv(vec=weights_H[i])
        x = img_aug[2, idx:idx + kernel_width, :]
        perm_2 = list(range(x.ndim))
        perm_2[0] = 1
        perm_2[1] = 0
        out_1[2, i, :] = x.transpose(perm=perm_2).mv(vec=weights_H[i])
    out_1_aug = paddle.empty(shape=[in_C, out_H, in_W + sym_len_Ws +
        sym_len_We], dtype='float32')
    start_7 = out_1_aug.shape[2] + sym_len_Ws if sym_len_Ws < 0 else sym_len_Ws
    paddle.assign(out_1, output=paddle.slice(out_1_aug, [2], [start_7], [
        start_7 + in_W]))
    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = paddle.arange(start=sym_patch.shape[2] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=2, index=inv_idx)
    start_8 = out_1_aug.shape[2] + 0 if 0 < 0 else 0
    paddle.assign(sym_patch_inv, output=paddle.slice(out_1_aug, [2], [
        start_8], [start_8 + sym_len_Ws]))
    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = paddle.arange(start=sym_patch.shape[2] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=2, index=inv_idx)
    start_9 = out_1_aug.shape[2
        ] + sym_len_Ws + in_W if sym_len_Ws + in_W < 0 else sym_len_Ws + in_W
    paddle.assign(sym_patch_inv, output=paddle.slice(out_1_aug, [2], [
        start_9], [start_9 + sym_len_We]))
    out_2 = paddle.empty(shape=[in_C, out_H, out_W], dtype='float32')
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(vec=
            weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(vec=
            weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(vec=
            weights_W[i])
    return out_2


def imresize_np(img, scale, antialiasing=True):
    img = paddle.to_tensor(data=img)
    in_H, in_W, in_C = img.shape
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    img_aug = paddle.empty(shape=[in_H + sym_len_Hs + sym_len_He, in_W,
        in_C], dtype='float32')
    start_10 = img_aug.shape[0] + sym_len_Hs if sym_len_Hs < 0 else sym_len_Hs
    paddle.assign(img, output=paddle.slice(img_aug, [0], [start_10], [
        start_10 + in_H]))
    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = paddle.arange(start=sym_patch.shape[0] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=0, index=inv_idx)
    start_11 = img_aug.shape[0] + 0 if 0 < 0 else 0
    paddle.assign(sym_patch_inv, output=paddle.slice(img_aug, [0], [
        start_11], [start_11 + sym_len_Hs]))
    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = paddle.arange(start=sym_patch.shape[0] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=0, index=inv_idx)
    start_12 = img_aug.shape[0
        ] + sym_len_Hs + in_H if sym_len_Hs + in_H < 0 else sym_len_Hs + in_H
    paddle.assign(sym_patch_inv, output=paddle.slice(img_aug, [0], [
        start_12], [start_12 + sym_len_He]))
    out_1 = paddle.empty(shape=[out_H, in_W, in_C], dtype='float32')
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        x = img_aug[idx:idx + kernel_width, :, 0]
        perm_3 = list(range(x.ndim))
        perm_3[0] = 1
        perm_3[1] = 0
        out_1[i, :, 0] = x.transpose(perm=perm_3).mv(vec=weights_H[i])
        x = img_aug[idx:idx + kernel_width, :, 1]
        perm_4 = list(range(x.ndim))
        perm_4[0] = 1
        perm_4[1] = 0
        out_1[i, :, 1] = x.transpose(perm=perm_4).mv(vec=weights_H[i])
        x = img_aug[idx:idx + kernel_width, :, 2]
        perm_5 = list(range(x.ndim))
        perm_5[0] = 1
        perm_5[1] = 0
        out_1[i, :, 2] = x.transpose(perm=perm_5).mv(vec=weights_H[i])
    out_1_aug = paddle.empty(shape=[out_H, in_W + sym_len_Ws + sym_len_We,
        in_C], dtype='float32')
    start_13 = out_1_aug.shape[1
        ] + sym_len_Ws if sym_len_Ws < 0 else sym_len_Ws
    paddle.assign(out_1, output=paddle.slice(out_1_aug, [1], [start_13], [
        start_13 + in_W]))
    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = paddle.arange(start=sym_patch.shape[1] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=1, index=inv_idx)
    start_14 = out_1_aug.shape[1] + 0 if 0 < 0 else 0
    paddle.assign(sym_patch_inv, output=paddle.slice(out_1_aug, [1], [
        start_14], [start_14 + sym_len_Ws]))
    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = paddle.arange(start=sym_patch.shape[1] - 1, end=-1, step=-1
        ).astype(dtype='int64')
    sym_patch_inv = sym_patch.index_select(axis=1, index=inv_idx)
    start_15 = out_1_aug.shape[1
        ] + sym_len_Ws + in_W if sym_len_Ws + in_W < 0 else sym_len_Ws + in_W
    paddle.assign(sym_patch_inv, output=paddle.slice(out_1_aug, [1], [
        start_15], [start_15 + sym_len_We]))
    out_2 = paddle.empty(shape=[out_H, out_W, in_C], dtype='float32')
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(vec=
            weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(vec=
            weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(vec=
            weights_W[i])
    return out_2.numpy()


def filtering(img_gray, r, eps):
    img = np.copy(img_gray)
    H = 1 / np.square(r) * np.ones([r, r])
    meanI = scipy.ndimage.correlate(img, H, mode='nearest')
    var = scipy.ndimage.correlate(img * img, H, mode='nearest') - meanI * meanI
    a = var / (var + eps)
    b = meanI - a * meanI
    meana = scipy.ndimage.correlate(a, H, mode='nearest')
    meanb = scipy.ndimage.correlate(b, H, mode='nearest')
    output = meana * img + meanb
    return output


def guided_filter(img_LR, r=5, eps=0.01):
    img = np.copy(img_LR)
    for i in range(3):
        img[:, :, i] = filtering(img[:, :, i], r, eps)
    return img
