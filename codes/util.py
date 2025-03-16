from datetime import datetime
import os
import math
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import glob
import torch
import cv2
import torch.nn as nn
import torch.nn.init as init
from torchvision.utils import make_grid
import torch.nn.functional as F
import yaml


####################
# Files & IO
####################

def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    return yaml_cfg


###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def get_public_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


####################
# image processing
# process on numpy image
####################

def show_tensor_image(tensor: torch.Tensor, title: str = ""):
    """
    Show a torch tensor image (CHW) using matplotlib.

    Parameters:
    - tensor: torch.Tensor, shape (C, H, W)
    - title: str, optional, the title of the plot
    """
    # Check if tensor is in the expected shape (C, H, W)
    if tensor.ndimension() != 3 or tensor.size(0) not in [1, 3]:
        raise ValueError(
            "Expected a 3D tensor with shape (C, H, W) where C is 1 or 3.")

    # Convert the tensor to numpy array and transpose it to (H, W, C)
    image = tensor.numpy().transpose((1, 2, 0))

    # If the image has only one channel (grayscale), remove the last dimension
    if image.shape[2] == 1:
        image = image[:, :, 0]

    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()


def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
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
    """horizontal flip OR rotate (0, 90, 180, 270 degrees) with flows"""
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


def channel_convert(in_c, tar_type, img_list):
    """conversion among BGR, gray and y"""
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    elif in_c == 3 and tar_type == 'RGB':  # bgr to rgb
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    """img_in: Numpy, HWC or HW"""
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


####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (
                absx <= 2)).type_as(
            absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width,
                                 :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width,
                                 :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width,
                                 :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx +
                                   kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx +
                                   kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx +
                                   kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width,
                                 :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width,
                                 :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width,
                                 :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx +
                                   kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx +
                                   kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx +
                                   kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


def set_require_grad(net: torch.nn.Module | list, requires_grad: bool):
    if isinstance(net, list):
        for n in net:
            set_require_grad(n, requires_grad)
        return
    for param in net.parameters():
        param.requires_grad = requires_grad


def quant(x: torch.Tensor):
    output = torch.clamp(x, 0, 1)
    output = (output * 255.0).round() / 255.0
    return output


def tensor2float_bchw(x: torch.Tensor):
    if len(x.shape) == 3:
        x = x.view(1, *x.shape)
    return x.float()


def clip_grad_norm(nets, norm: float):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=norm)


def calculate_psnr(
        img1: torch.Tensor | np.ndarray,
        img2: torch.Tensor | np.ndarray,
) -> float:
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim2d(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """

    :param img1: HxW or CxHxW or NxCxHxW
    :param img2:
    :return:
    """
    img1, img2 = np.squeeze(img1), np.squeeze(img2)
    if img1.ndim == 2:
        return calculate_ssim2d(img1, img2)
    elif img1.ndim == 3:
        ssim = 0
        for i in range(img1.shape[2]):
            ssim += calculate_ssim2d(img1[:, :, i], img2[:, :, i])
        return ssim / img1.shape[2]
    elif img1.ndim == 4:
        ssim = 0
        for i in range(img1.shape[3]):
            ssim += calculate_ssim(img1[:, :, :, i], img2[:, :, :, i])
        return ssim / img1.shape[3]
    else:
        raise ValueError("dimensions error")


def initialize_weights(net_l, scale: float = 1):
    """
    初始化网络参数
    :param net_l: 要初始化的网络
    :param scale: 倍数
    :return: none
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            # kaiming 正态分布
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def get_day():
    now = datetime.now()
    day = now.day
    return day


def get_day_and_hour():
    now = datetime.now()
    day = now.day
    hour = now.hour
    return day, hour


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        os.rename(path, new_name)
    os.makedirs(path)


def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [
            v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list
        ]


def tensor2img(tensor: torch.Tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().detach().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(
            tensor, nrow=int(math.sqrt(n_img)), normalize=False
        ).numpy()
        img_np = np.transpose(
            img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(
            img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def calculate_psnr_ssim(img1, img2, scale, min_max=(0, 1), only_y=False):
    img1, img2 = tensor2img(img1, np.uint8, min_max), tensor2img(
        img2, np.uint8, min_max
    )
    if only_y:
        img1 = rgb2ycbcr(img1, only_y=True)
        img2 = rgb2ycbcr(img2, only_y=True)
    img1, img2 = crop_border([img1, img2], scale)
    p = calculate_psnr(img1, img2)
    s = calculate_ssim(img1, img2)
    return p, s


def model_load_weight(net: nn.Module, weight_path: str, strict: bool = True):
    def check_param_size(param, loaded_param):
        return param.size() == loaded_param.size()

    def remove_unexpected_perfix(param_name: str, prefix: str):
        if param_name.startswith(prefix):
            return param_name[len(prefix):]
        else:
            return param_name

    unexpected_prefix = ["module."]
    loaded_state_dict = torch.load(weight_path)
    target_state_dict = net.state_dict()
    for param in loaded_state_dict:
        for prefix in unexpected_prefix:
            param_name = remove_unexpected_perfix(param, prefix)
            if param_name in target_state_dict:
                if check_param_size(target_state_dict[param_name], loaded_state_dict[param]):
                    target_state_dict[param_name].copy_(
                        loaded_state_dict[param])
    net.load_state_dict(target_state_dict, strict=strict)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(
                unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,
                                                                                                           w + 2 * f * pad,
                                                                                                           h + 2 * f * pad)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(
                unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c,
                                                                                                                 w + 2 * f * pad,
                                                                                                                 h + 2 * f * pad)


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(
            c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        before_shuffle = x.view(
            b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def ll_r3(x: torch.Tensor, denoised: torch.Tensor, net: nn.Module, r3_factor: float = 0.16, r3_num: int = 8, p: int = 0,
          st: int = 2) -> torch.Tensor:
    """random replacement Refinement with ratio r3_factor for large model

    Note: 
        This module is only used in eval, not in train. val will take r3_num times longer than train.
    Args:
        x(torch.Tensor): input tensor.BCHW,B=1
        net(nn.Module): model to eval
        r3_factor (float, optional): the ratio of radnom replace. Defaults to 0.16.
        r3_num (int, optional): the number of r3 times. Defaults to 8.
    Output: BCHW
    """
    assert st <= r3_num and st >= 2, "value error!!"
    b, c, h, w = x.shape
    temp_input = denoised.expand(r3_num, -1, -1, -1)
    x = x.expand(r3_num, -1, -1, -1)
    indices = torch.zeros(r3_num, c, h, w, dtype=torch.bool, device=x.device)
    for t in range(r3_num):
        indices[t] = (torch.rand(1, h, w) < r3_factor).repeat(3, 1, 1)
    temp_input = temp_input.clone()
    temp_input[indices] = x[indices]
    temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
    res = torch.zeros_like(temp_input)
    with torch.no_grad():
        if p == 0:
            res[:r3_num // 2, ...] = net(temp_input[:r3_num // 2, ...])
            res[r3_num // 2:, ...] = net(temp_input[r3_num // 2:, ...])
        else:
            res[:r3_num //
                2, ...] = net(temp_input[:r3_num // 2, ...])[:, :, p:-p, p:-p]
            res[r3_num // 2:, ...] = net(temp_input[r3_num //
                                                    2:, ...])[:, :, p:-p, p:-p]
    return torch.mean(res, dim=0).unsqueeze(0)


def r3(x: torch.Tensor, denoised: torch.Tensor, net: nn.Module, r3_factor: float = 0.16, r3_num: int = 8,
       p: int = 0) -> torch.Tensor:
    """random replacement Refinement with ratio r3_factor

    Note: 
        This module is only used in eval, not in train. val will take r3_num times longer than train.
    Args:
        x(torch.Tensor): input tensor.BCHW,B=1
        net(nn.Module): model to eval
        r3_factor (float, optional): the ratio of radnom replace. Defaults to 0.16.
        r3_num (int, optional): the number of r3 times. Defaults to 8.
    Output: BCHW
    """
    b, c, h, w = x.shape
    temp_input = denoised.expand(r3_num, -1, -1, -1)
    x = x.expand(r3_num, -1, -1, -1).to(dtype=torch.float32)
    indices = torch.zeros(r3_num, c, h, w, dtype=torch.bool, device=x.device)
    for t in range(r3_num):
        indices[t] = (torch.rand(1, h, w) < r3_factor).repeat(3, 1, 1)
    temp_input = temp_input.clone()
    temp_input[indices] = x[indices]
    temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
    with torch.no_grad():
        if p == 0:
            denoised = net(temp_input)
        else:
            denoised = net(temp_input)[:, :, p:-p, p:-p]
    return torch.mean(denoised, dim=0).unsqueeze(0)


def get_state_dict(net: nn.Module | nn.DataParallel) -> dict:
    if isinstance(net, nn.DataParallel) or isinstance(net, nn.parallel.DistributedDataParallel):
        state_dict = net.module.state_dict()
    elif isinstance(net, nn.Module):
        state_dict = net.state_dict()
    return state_dict


def pd_down(x: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    b, c, h, w = x.shape
    x_down = F.pixel_unshuffle(x, pd_factor)
    out = x_down.view(b, c, pd_factor, pd_factor, h // pd_factor, w // pd_factor).permute(
        0, 2, 3, 1, 4, 5).reshape(b * pd_factor * pd_factor, c, h // pd_factor, w // pd_factor)
    return out


def pd_up(out: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    b, c, h, w = out.shape
    # Reshape the output tensor to its original shape after pixel unshuffle
    x_down = out.view(b // (pd_factor ** 2), pd_factor, pd_factor, c, h,
                      w).permute(0, 3, 1, 2, 4, 5)
    x_down = x_down.reshape(b // (pd_factor ** 2), c *
                            pd_factor * pd_factor, h, w)
    # Use pixel shuffle to upsample the tensor
    x_up = F.pixel_shuffle(x_down, pd_factor)
    return x_up


def bm3d(noisy: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(noisy, torch.Tensor):
        noisy = tensor2img(noisy, out_type=np.float32, min_max=(0, 255))
    from scipy.fftpack import dct, idct
    sigma = 25  # variance of the noise

    lamb2d = 2.0

    lamb3d = 2.7

    Step1_ThreDist = 2500  # threshold distance

    Step1_MaxMatch = 16  # max matched blocks

    Step1_BlockSize = 8

    Step1_spdup_factor = 3  # pixel jump for new reference block

    Step1_WindowSize = 39  # search window size

    Step2_ThreDist = 400

    Step2_MaxMatch = 32

    Step2_BlockSize = 8

    Step2_spdup_factor = 3

    Step2_WindowSize = 39

    Kaiser_Window_beta = 2.0

    def Initialization(Img, BlockSize, Kaiser_Window_beta):
        """
        Initialize the image, weight and Kaiser window

        Return:
            InitImg & InitWeight: zero-value Img.shape matrices 
                    InitKaiser: (BlockSize * BlockSize) Kaiser window 
        """

        InitImg = np.zeros(Img.shape, dtype=float)

        InitWeight = np.zeros(Img.shape, dtype=float)

        Window = np.matrix(np.kaiser(BlockSize, Kaiser_Window_beta))

        InitKaiser = np.array(Window.T * Window)

        return InitImg, InitWeight, InitKaiser

    def SearchWindow(Img, RefPoint, BlockSize, WindowSize):
        """ 
        Find the search window whose center is reference block in *Img*

        Note that the center of SearchWindow is not always the reference block because of the border   

        Return:
            (2 * 2) array of left-top and right-bottom coordinates in search window
        """

        if BlockSize >= WindowSize:
            print('Error: BlockSize is smaller than WindowSize.\n')

            exit()

        Margin = np.zeros((2, 2), dtype=int)

        Margin[0, 0] = max(
            0, RefPoint[0] + int((BlockSize - WindowSize) / 2))  # left-top x

        Margin[0, 1] = max(
            0, RefPoint[1] + int((BlockSize - WindowSize) / 2))  # left-top y

        Margin[1, 0] = Margin[0, 0] + WindowSize  # right-bottom x

        Margin[1, 1] = Margin[0, 1] + WindowSize  # right-bottom y

        if Margin[1, 0] >= Img.shape[0]:
            Margin[1, 0] = Img.shape[0] - 1

            Margin[0, 0] = Margin[1, 0] - WindowSize

        if Margin[1, 1] >= Img.shape[1]:
            Margin[1, 1] = Img.shape[1] - 1

            Margin[0, 1] = Margin[1, 1] - WindowSize

        return Margin

    def dct2D(A):
        """
        2D discrete cosine transform (DCT)
        """

        return dct(dct(A, axis=0, norm='ortho'), axis=1, norm='ortho')

    def idct2D(A):
        """
        inverse 2D discrete cosine transform
        """

        return idct(idct(A, axis=0, norm='ortho'), axis=1, norm='ortho')

    def PreDCT(Img, BlockSize):
        """
        Do discrete cosine transform (2D transform) for each block in *Img* to reduce the complexity of 
        applying transforms

        Return:
            BlockDCT_all: 4-dimensional array whose first two dimensions correspond to the block's 
                        position and last two correspond to the DCT array of the block
        """

        BlockDCT_all = np.zeros((Img.shape[0] - BlockSize, Img.shape[1] - BlockSize, BlockSize, BlockSize),
                                dtype=float)

        for i in range(BlockDCT_all.shape[0]):

            for j in range(BlockDCT_all.shape[1]):
                Block = Img[i:i + BlockSize, j:j + BlockSize]

                BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))

        return BlockDCT_all

    def Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, ThreDist, MaxMatch, WindowSize):
        """
        Find blocks similar to the reference one in *noisyImg* based on *BlockDCT_all*

        Note that the distance computing is chosen from original paper rather than the analysis one

        Return:
            BlockPos: array of blocks' position (left-top point)
            BlockGroup: 3-dimensional array whose last two dimensions correspond to the DCT array of 
                        the block
        """

        # initialization

        WindowLoc = SearchWindow(noisyImg, RefPoint, BlockSize, WindowSize)

        # number of searched blocks
        Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2

        BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)

        BlockGroup = np.zeros(
            (Block_Num_Searched, BlockSize, BlockSize), dtype=float)

        Dist = np.zeros(Block_Num_Searched, dtype=float)

        RefDCT = BlockDCT_all[RefPoint[0], RefPoint[1], :, :]

        match_cnt = 0

        # Block searching and similarity (distance) computing

        for i in range(WindowSize - BlockSize + 1):

            for j in range(WindowSize - BlockSize + 1):

                SearchedDCT = BlockDCT_all[WindowLoc[0,
                                                     0] + i, WindowLoc[0, 1] + j, :, :]

                dist = Step1_ComputeDist(RefDCT, SearchedDCT)

                if dist < ThreDist:
                    BlockPos[match_cnt, :] = [
                        WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]

                    BlockGroup[match_cnt, :, :] = SearchedDCT

                    Dist[match_cnt] = dist

                    match_cnt += 1

        #    if match_cnt == 1:
        #
        #        print('WARNING: no similar blocks founded for the reference block {} in basic estimate.\n'\
        #              .format(RefPoint))

        if match_cnt <= MaxMatch:

            # less than MaxMatch similar blocks founded, return similar blocks

            BlockPos = BlockPos[:match_cnt, :]

            BlockGroup = BlockGroup[:match_cnt, :, :]

        else:

            # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks

            # indices of MaxMatch smallest distances
            idx = np.argpartition(Dist[:match_cnt], MaxMatch)

            BlockPos = BlockPos[idx[:MaxMatch], :]

            BlockGroup = BlockGroup[idx[:MaxMatch], :]

        return BlockPos, BlockGroup

    def Step1_ComputeDist(BlockDCT1, BlockDCT2):
        """
        Compute the distance of two DCT arrays *BlockDCT1* and *BlockDCT2* 
        """

        if BlockDCT1.shape != BlockDCT1.shape:

            print(
                'ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')

            return

        elif BlockDCT1.shape[0] != BlockDCT1.shape[1]:

            print('ERROR: DCT Block is not square in step1 computing distance.\n')

            return

        BlockSize = BlockDCT1.shape[0]

        if sigma > 40:
            ThreValue = lamb2d * sigma

            BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)

            BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)

        return np.linalg.norm(BlockDCT1 - BlockDCT2) ** 2 / (BlockSize ** 2)

    def Step1_3DFiltering(BlockGroup):
        """
        Do collaborative hard-thresholding which includes 3D transform, noise attenuation through 
        hard-thresholding and inverse 3D transform

        Return:
            BlockGroup
        """

        ThreValue = lamb3d * sigma

        nonzero_cnt = 0

        # since 2D transform has been done, we do 1D transform, hard-thresholding and inverse 1D
        # transform, the inverse 2D transform is left in aggregation processing

        for i in range(BlockGroup.shape[1]):

            for j in range(BlockGroup.shape[2]):
                ThirdVector = dct(BlockGroup[:, i, j], norm='ortho')  # 1D DCT

                ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.

                nonzero_cnt += np.nonzero(ThirdVector)[0].size

                BlockGroup[:, i, j] = list(idct(ThirdVector, norm='ortho'))

        return BlockGroup, nonzero_cnt

    def Step1_Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt):
        """
        Compute the basic estimate of the true-image by weighted averaging all of the obtained 
        block-wise estimates that are overlapping

        Note that the weight is set accroding to the original paper rather than the BM3D analysis one
        """

        if nonzero_cnt < 1:

            BlockWeight = 1.0 * basicKaiser

        else:

            BlockWeight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser

        for i in range(BlockPos.shape[0]):
            basicImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1],
                     BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] \
                += BlockWeight * idct2D(BlockGroup[i, :, :])

            basicWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1],
                        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] += BlockWeight

    def BM3D_Step1(noisyImg):
        """
        Give the basic estimate after grouping, collaborative filtering and aggregation

        Return:
            basic estimate basicImg
        """

        # preprocessing

        BlockSize = Step1_BlockSize

        ThreDist = Step1_ThreDist

        MaxMatch = Step1_MaxMatch

        WindowSize = Step1_WindowSize

        spdup_factor = Step1_spdup_factor

        basicImg, basicWeight, basicKaiser = Initialization(
            noisyImg, BlockSize, Kaiser_Window_beta)

        BlockDCT_all = PreDCT(noisyImg, BlockSize)

        # block-wise estimate with speed-up factor

        for i in range(int((noisyImg.shape[0] - BlockSize) / spdup_factor) + 2):

            for j in range(int((noisyImg.shape[1] - BlockSize) / spdup_factor) + 2):
                RefPoint = [min(spdup_factor * i, noisyImg.shape[0] - BlockSize - 1),
                            min(spdup_factor * j, noisyImg.shape[1] - BlockSize - 1)]

                BlockPos, BlockGroup = Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize,
                                                      ThreDist, MaxMatch, WindowSize)

                BlockGroup, nonzero_cnt = Step1_3DFiltering(BlockGroup)

                Step1_Aggregation(BlockGroup, BlockPos, basicImg,
                                  basicWeight, basicKaiser, nonzero_cnt)

        basicWeight = np.where(basicWeight == 0, 1, basicWeight)

        basicImg[:, :] /= basicWeight[:, :]

        #    basicImg = (np.matrix(basicImg, dtype=int)).astype(np.uint8)

        return basicImg

    def Step2_Grouping(basicImg, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize,
                       BlockDCT_basic, BlockDCT_noisy):
        """
        Similar to Step1_Grouping, find the similar blocks to the reference one from *basicImg*

        Return:
                    BlockPos: array of similar blocks' position (left-top point)
            BlockGroup_basic: 3-dimensional array standing for the stacked blocks similar to the 
                            reference one from *basicImg* after 2D DCT
            BlockGroup_noisy: the stacked blocks from *noisyImg* corresponding to BlockGroup_basic 
        """

        # initialization (same as Step1)

        WindowLoc = SearchWindow(basicImg, RefPoint, BlockSize, WindowSize)

        Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2

        BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)

        BlockGroup_basic = np.zeros(
            (Block_Num_Searched, BlockSize, BlockSize), dtype=float)

        BlockGroup_noisy = np.zeros(
            (Block_Num_Searched, BlockSize, BlockSize), dtype=float)

        Dist = np.zeros(Block_Num_Searched, dtype=float)

        match_cnt = 0

        # Block searching and similarity (distance) computing
        # Note the distance computing method is different from that of Step1

        for i in range(WindowSize - BlockSize + 1):

            for j in range(WindowSize - BlockSize + 1):

                SearchedPoint = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]

                dist = Step2_ComputeDist(
                    basicImg, RefPoint, SearchedPoint, BlockSize)

                if dist < ThreDist:
                    BlockPos[match_cnt, :] = SearchedPoint

                    Dist[match_cnt] = dist

                    match_cnt += 1

        #    if match_cnt == 1:
        #
        #        print('WARNING: no similar blocks founded for the reference block {} in final estimate.\n'\
        #              .format(RefPoint))

        if match_cnt <= MaxMatch:

            # less than MaxMatch similar blocks founded, return similar blocks

            BlockPos = BlockPos[:match_cnt, :]

        else:

            # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks

            # indices of MaxMatch smallest distances
            idx = np.argpartition(Dist[:match_cnt], MaxMatch)

            BlockPos = BlockPos[idx[:MaxMatch], :]

        for i in range(BlockPos.shape[0]):
            SimilarPoint = BlockPos[i, :]

            BlockGroup_basic[i, :, :] = BlockDCT_basic[SimilarPoint[0],
                                                       SimilarPoint[1], :, :]

            BlockGroup_noisy[i, :, :] = BlockDCT_noisy[SimilarPoint[0],
                                                       SimilarPoint[1], :, :]

        BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :, :]

        BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :, :]

        return BlockPos, BlockGroup_basic, BlockGroup_noisy

    def Step2_ComputeDist(img, Point1, Point2, BlockSize):
        """
        Compute distance between blocks whose left-top margins' coordinates are *Point1* and *Point2*        
        """

        Block1 = (img[Point1[0]:Point1[0] + BlockSize, Point1[1]                  :Point1[1] + BlockSize]).astype(np.float64)

        Block2 = (img[Point2[0]:Point2[0] + BlockSize, Point2[1]                  :Point2[1] + BlockSize]).astype(np.float64)

        return np.linalg.norm(Block1 - Block2) ** 2 / (BlockSize ** 2)

    def Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy):
        """
        Do collaborative Wiener filtering and here we choose 2D DCT + 1D DCT as the 3D transform which 
        is the same with the 3D transform in hard-thresholding filtering

        Note that the Wiener weight is set accroding to the BM3D analysis paper rather than the original 
        one

        Return:
        BlockGroup_noisy & WienerWeight
        """

        Weight = 0

        coef = 1.0 / BlockGroup_noisy.shape[0]

        for i in range(BlockGroup_noisy.shape[1]):

            for j in range(BlockGroup_noisy.shape[2]):
                Vec_basic = dct(BlockGroup_basic[:, i, j], norm='ortho')

                Vec_noisy = dct(BlockGroup_noisy[:, i, j], norm='ortho')

                Vec_value = Vec_basic ** 2 * coef

                Vec_value /= (Vec_value + sigma ** 2)  # pixel weight

                Vec_noisy *= Vec_value

                Weight += np.sum(Vec_value)
                #            for k in range(BlockGroup_noisy.shape[0]):
                #
                #                Value = Vec_basic[k]**2 * coef
                #
                #                Value /= (Value + sigma**2) # pixel weight
                #
                #                Vec_noisy[k] = Vec_noisy[k] * Value
                #
                #                Weight += Value

                BlockGroup_noisy[:, i, j] = list(idct(Vec_noisy, norm='ortho'))

        if Weight > 0:

            WienerWeight = 1. / (sigma ** 2 * Weight)

        else:

            WienerWeight = 1.0

        return BlockGroup_noisy, WienerWeight

    def Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, finalKaiser):
        """
        Compute the final estimate of the true-image by aggregating all of the obtained local estimates 
        using a weighted average 
        """

        BlockWeight = WienerWeight * finalKaiser

        for i in range(BlockPos.shape[0]):
            finalImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1],
                     BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] \
                += BlockWeight * idct2D(BlockGroup_noisy[i, :, :])

            finalWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1],
                        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] += BlockWeight

    def BM3D_Step2(basicImg, noisyImg):
        """
        Give the final estimate after grouping, Wiener filtering and aggregation

        Return:
            final estimate finalImg
        """

        # parameters setting

        BlockSize = Step2_BlockSize

        ThreDist = Step2_ThreDist

        MaxMatch = Step2_MaxMatch

        WindowSize = Step2_WindowSize

        spdup_factor = Step2_spdup_factor

        finalImg, finalWeight, finalKaiser = Initialization(
            basicImg, BlockSize, Kaiser_Window_beta)

        BlockDCT_noisy = PreDCT(noisyImg, BlockSize)

        BlockDCT_basic = PreDCT(basicImg, BlockSize)

        # block-wise estimate with speed-up factor

        for i in range(int((basicImg.shape[0] - BlockSize) / spdup_factor) + 2):

            for j in range(int((basicImg.shape[1] - BlockSize) / spdup_factor) + 2):
                RefPoint = [min(spdup_factor * i, basicImg.shape[0] - BlockSize - 1),
                            min(spdup_factor * j, basicImg.shape[1] - BlockSize - 1)]

                BlockPos, BlockGroup_basic, BlockGroup_noisy = Step2_Grouping(basicImg, noisyImg,
                                                                              RefPoint, BlockSize,
                                                                              ThreDist, MaxMatch,
                                                                              WindowSize,
                                                                              BlockDCT_basic,
                                                                              BlockDCT_noisy)

                BlockGroup_noisy, WienerWeight = Step2_3DFiltering(
                    BlockGroup_basic, BlockGroup_noisy)

                Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight,
                                  finalKaiser)

        finalWeight = np.where(finalWeight == 0, 1, finalWeight)

        finalImg[:, :] /= finalWeight[:, :]

        #    finalImg = (np.matrix(finalImg, dtype=int)).astype(np.uint8)

        return finalImg

    basic_img = BM3D_Step1(noisy)
    basic_img_uint = np.zeros(noisy.shape)
    cv2.normalize(basic_img, basic_img_uint, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    final_img = BM3D_Step2(basic_img, noisy)
    basic_img_uint = basic_img_uint.astype(np.uint8)
    cv2.normalize(final_img, final_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    final_img = final_img.astype(np.uint8)
    return final_img


def blur(img: torch.Tensor | np.ndarray, kernel_size: int = 3, sigma: float = 1.5) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = tensor2img(img, out_type=np.uint8, min_max=(0, 255))
    if isinstance(img, np.ndarray):
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return img


def add_noise(img: torch.Tensor | np.ndarray, noise_type: str = 'gaussian', sigma: float = 25) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = tensor2img(img, out_type=np.uint8, min_max=(0, 255))
    if isinstance(img, np.ndarray):
        if noise_type == 'gaussian':
            img = img + np.random.normal(0, sigma, img.shape)
        elif noise_type == 'poisson':
            img = img + np.random.poisson(img / 255.0 * sigma) / sigma * 255.0
        elif noise_type == 's&p':
            img = img + np.random.normal(0, sigma, img.shape)
        elif noise_type == 'speckle':
            img = img + np.random.normal(0, sigma, img.shape)
    return img


def random_resize(img: torch.Tensor | np.ndarray, scale: float = 0.5) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = tensor2img(img, out_type=np.uint8, min_max=(0, 255))
    if isinstance(img, np.ndarray):
        inter = np.random.choice(
            [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST])
        img = cv2.resize(
            img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=inter)
    return img


def jepg_compress(img: torch.Tensor | np.ndarray, quality: int = 50) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = tensor2img(img, out_type=np.uint8, min_max=(0, 255))
    if isinstance(img, np.ndarray):
        img = cv2.imencode(
            '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img
