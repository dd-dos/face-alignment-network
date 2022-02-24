from __future__ import print_function

import os
import shutil
import numpy as np
import random
import math
import cv2
from skimage import io

import torch
import torch.utils.data as data
# from torch.utils.serialization import load_lua

# from utils.utils import *
from PIL import Image
from utils.imutils import *
from utils.transforms import *
import opts
import logging
import torch.utils
import matplotlib.pyplot as plt

import torchvision.transforms as T
import random

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class W300LP(data.Dataset):
    def __init__(self, data_path, model_config, split):
        self.nParts = 68
        self.img_folder = data_path
        self.split = split
        self.is_train = True if self.split == 'train' else False

        self.transformer = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])

        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = model_config['scale_factor']
        self.rot_factor = model_config['rot_factor']
        self.mean, self.std = torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])

    def _getDataFaces(self, is_train):
        '''
        According to latest check: train size = 115118
                                   val size = 7332
        '''
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        base_dir = os.path.join(self.img_folder, 'landmarks')
        dirs = os.listdir(base_dir)
        lines = []
        vallines = []
        for d in dirs:
            files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.mat')]
            
            for f in files:
                if f.find('test') == -1:
                    lines.append(f) 
                else:
                    vallines.append(f)
        if is_train:
            logger.info('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            logger.info('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, center, scale, reference_scale = self.generateSampleFace(index)
        self.pts, self.center, self.scale, self.reference_scale = pts, center, scale, reference_scale
        if self.is_train:
            return inp, out
        else:
            meta = {'index': index, 'center': center, 'scale': scale, 'pts': pts, 'reference_scale': reference_scale}
            return inp, out, meta

    def generateSampleFace(self, idx):
        #landmark
        pts_path = os.path.join(self.img_folder, 'landmarks', self.anno[idx].split('_')[0], self.anno[idx][:-8] + '.mat')
        pts = scipy.io.loadmat(pts_path)['pt3d']
        #image
        sf = self.scale_factor
        rf = self.rot_factor

        imagepath = os.path.join(self.img_folder, self.anno[idx].split('_')[0], self.anno[idx][:-4] + '.jpg')

        if self.is_train:
            img = load_image(imagepath, augment=True)
        else:
            img = load_image(imagepath, augment=False)
        
        img = self.transformer(img)

        height = img.size(1)
        width = img.size(2)
        hw = max(width, height)
        #consider the landmarks are mainly distributed in the lower half face, so it needs move the center to some lower along height direction
        center = torch.FloatTensor(( float(width*1.0/2), float(height*1.0/2 + height*0.12) ))
        reference_scale = torch.tensor(200.0)
        #we hope face for train to be larger than in raw image, so edge will be short by 0.8 ration
        scale_x = hw*0.8 / reference_scale
        scale_y = hw*0.8 / reference_scale
        scale = torch.FloatTensor(( scale_x, scale_y ))

        rotate = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] 

        if self.is_train:
            scale[0] = scale[0] * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            scale[1] = scale[0]
            
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
                center[0] = img.size(2) - center[0]

        inp = crop(img, center, scale, reference_scale, [256, 256], rot=rotate)

        tpts = pts.copy()
        out = torch.zeros(self.nParts, 64, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], center, scale, reference_scale, [64, 64], rot=rotate))
                out[i] = draw_labelmap(out[i], tpts[i], sigma=1)

        return inp, out, pts, center, scale, reference_scale
        # return inp, out, tpts, center, scale, reference_scale


def tensor_to_img(normalized_img):
    res = normalized_img.numpy()*255
    res = res.transpose(1,2,0).astype(np.uint8)

    return res

def landmark_check(data):
    imgs, heatmaps = data
    for i in range(len(imgs)):
        img = (imgs[i].numpy().transpose(1,2,0)*255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        heatmap = heatmaps[i]
        out = heatmap.unsqueeze(0)
        center = torch.tensor([128,128*1.12])
        scale_x = 256/200
        scale_y = 256/200
        reference_scale = 200
        pts, pts_img = get_preds_fromhm(out, [center], [[scale_x, scale_y]], [reference_scale])
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
        for j in range(pts.shape[0]):
            _pts = pts[j]
            cv2.circle(img, (_pts[0], _pts[1]),2,(0,255,0), -1, 8)
        cv2.imwrite('./overfit_imgs/{}.jpg'.format(i), img)


def landmark_check_image(data):
    img, heatmap = data
    
    img = (img.numpy().transpose(1,2,0)*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = heatmap.unsqueeze(0)
    center = torch.tensor([128,128*1.12])
    scale_x = 256/200
    scale_y = 256/200
    reference_scale = 200
    pts, pts_img = get_preds_fromhm(out, [center], [[scale_x, scale_y]], [reference_scale])
    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
    for j in range(pts.shape[0]):
        _pts = pts[j]
        cv2.circle(img, (_pts[0], _pts[1]),2,(0,255,0), -1, 8)

    # cv2.imshow("", img)
    # cv2.waitKey(0)
    return img


if __name__=="__main__":

    import tqdm
    # loader = torch.utils.data.DataLoader(
    #     W300LP(args, 'train'),
    #     batch_size=args.train_batch,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True)

    data_path = 'datasets/300WLP'
    model_config = {
        'scale_factor': 0.3,
        'rot_factor': 90
    }
    loader = W300LP(data_path, model_config, "train")

    ####################################################################

    # for data in tqdm.tqdm(loader):
    #     img, heatmap = data[:2]
    #     img = landmark_check_image((img, heatmap))
    #     cv2.imshow("", img)
    #     key = cv2.waitKey(0)
    #     if key == 27: break

    ####################################################################