from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import os
import math
import cv2

import random

from .transforms import *
from .misc import *
import imgaug as ia
import imgaug.augmenters as iaa

# =============================================================================
# Helpful functions generating groundtruth labelmap
# =============================================================================

def gaussian(
        size=7, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(- \
                    (math.pow((j + 1 - center_x) / (sigma_horz * width), 2) / 2.0 \
                    + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0) \
                    )
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss    

def draw_labelmap(image, point, sigma):
    image = to_numpy(image)
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return to_torch(image)
    size = 6 * sigma + 1
    g = gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return to_torch(image)    

# =============================================================================
# Helpful display functions
# =============================================================================

aug = iaa.Sequential([
    iaa.Sometimes(0.5, 
        iaa.OneOf([
            iaa.imgcorruptlike.Brightness(severity=(1,3)),
            iaa.imgcorruptlike.Saturate(severity=[1, 3]),
            iaa.imgcorruptlike.Contrast(severity=1),
            iaa.ChangeColorTemperature((4000, 12000)),
            iaa.SigmoidContrast(gain=(3, 4), cutoff=(0.4, 0.6), per_channel=True),
            iaa.LogContrast(gain=(0.7, 1.4)),
            iaa.SigmoidContrast(gain= (3, 7), cutoff=(0.3, 0.6)),
            iaa.LinearContrast((0.3, 0.7)),
            iaa.RemoveSaturation((0.1, 0.2)),
        ]),
    ),

    iaa.Sometimes(0.9, 
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(1, 3)),
            iaa.AverageBlur(k=(3, 9)),
            iaa.MotionBlur(k=(3, 7)),
            iaa.imgcorruptlike.Pixelate(severity=(3, 5)),
            iaa.Pepper((0.01, 0.1)),
            iaa.AdditiveGaussianNoise(scale=(1, 10), per_channel=True),
            iaa.imgcorruptlike.SpeckleNoise(severity=(1, 2)),
            iaa.imgcorruptlike.JpegCompression(severity=(2, 4)),
            iaa.AveragePooling((1, 2)),
            iaa.pillike.FilterSmoothMore(),
            iaa.KMeansColorQuantization(n_colors=(150, 256))
        ])
    )
    # iaa.Sometimes(0.3, LightFlare()),
    # iaa.Sometimes(0.2, ParallelLight()),
    # iaa.Sometimes(0.01, SpotLight()),
    # iaa.Sometimes(0.2, RandomLine()),
    # iaa.Sometimes(0.2, BlobAndShadow()),
    # iaa.Sometimes(0.05, WarpTexture())
    ])

# transformer = T.Compose([
#                     T.ToTensor(),
#                     T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#                     ])

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()

    if img.max() > 1:
        img /= 255
    return img


def load_image(img_path, augment=False):
    assert os.path.isfile(img_path), "File path {} is not existed. Please check!".format(img_path)
    # H x W x C => C x H x W
    # return im_to_torch(scipy.misc.imread(img_path, mode='RGB'),aug=False)
    # return im_to_torch(Image.open(img_path), aug=aug)
    if random.random() < 0.5:
        img = cv2.imread(img_path)
        if img is None:
            raise TypeError("Image is None. Please check path: {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = Image.open(img_path)
        img = np.array(img)

    if augment:
        img = aug.augment(image=img)
    
    return img


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    print('%f %f' % (img.min(), img.max()))
    img = scipy.misc.imresize(
            img,
            (oheight, owidth)
        )
    
    img = im_to_torch(img)
    print('%f %f' % (img.min(), img.max()))
    return img


def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d


def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def imshow(img):
    npimg = im_to_numpy(img*255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis('off')


def show_joints(img, pts):
    imshow(img)

    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
    plt.axis('off')


def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp*0.5 + color_heatmap(target[n,p,:,:])*0.5
            out = torch.cat((out, tgt), 2)

        imshow(out)
        plt.show()


def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float)/255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0.5, 0.5, 0.5]), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return np.concatenate(batch_img)
