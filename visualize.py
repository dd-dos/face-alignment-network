import glob
import os
from skimage import io
import scipy.io
from PIL import Image
import cv2
import tqdm

def show(img=None, landmarks=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    if img is None:
        img = np.zeros((3,256,256))
        img[:,:,:] = 255
        img = img.transpose((1,2,0))
        
    plt.imshow(img)
    if landmarks is not None:
        plt.scatter(landmarks.T[0], landmarks.T[1], 10)
    plt.show()
    
def visualize_pure_train():
    landmark_folder_path = "./datasets/300WLP/landmarks"
    visualize_folder = "./datasets/300WLP_visualization"
    idx = 0
    for dtset in os.listdir(landmark_folder_path):
        if dtset.endswith('std'):
            files = os.listdir(os.path.join(landmark_folder_path, dtset))
            for flag, file in tqdm.tqdm(enumerate(files)):
                img_path = os.path.join("./datasets/300WLP", dtset, file[:-3] + "jpg")
                landmark_path = os.path.join(landmark_folder_path, dtset, file)
                img = cv2.imread(img_path)
                landmark = scipy.io.loadmat(landmark_path)['pt3d']
                for i in range(landmark.shape[0]):
                    pts = landmark[i]
                    img = cv2.circle(img, (pts[0], pts[1]),3,(0,255,0), -1, 8)
                cv2.imwrite('/home/pdd/Desktop/workspace/face-alignment/datasets/300WLP_visualization/{}.jpg'.format(idx),img)
                idx += 1
                if flag == 10000:
                    break


import torch
import torch.utils.data as data
# from torch.utils.serialization import load_lua

# from utils.utils import *
from PIL import Image
from utils.imutils import *
from utils.transforms import *
import opts
import logging
import random

class W300LP(data.Dataset):
    def __init__(self, split, type='rot'):
        self.nParts = 68
        self.type = type
        self.img_folder = "/home/pdd/Desktop/workspace/face-alignment/datasets/300WLP"
        self.split = split
        self.is_train = True if self.split == 'train' else False
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = 0.3
        self.rot_factor = 30
        self.mean, self.std = torch.tensor([0.5,0.5,0.5]), torch.tensor([0.229,0.224,0.225]) # taken from sig_ReID

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

        if self.type=='rot':
            dirs = ["/home/pdd/Desktop/workspace/face-alignment/datasets/300WLP/landmarks/300WLP-rot"]
        elif self.type=='std':
            dirs = ["/home/pdd/Desktop/workspace/face-alignment/datasets/300WLP/landmarks/300WLP-std"]

        for d in dirs:
            files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.mat')]
            # import ipdb; ipdb.set_trace()
            for f in files:
                if f.find('test') == -1:
                    lines.append(f) 
                else:
                    vallines.append(f)
        if is_train:
            logger.warning('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            logger.warning('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, center, scale, reference_scale = self.generateSampleFace(index)
        # import ipdb; ipdb.set_trace()
        self.pts, self.center, self.scale, self.reference_scale = pts, center, scale, reference_scale
        if self.is_train:
            return inp, out, pts
        else:
            meta = {'index': index, 'center': center, 'scale': scale, 'pts': pts, 'reference_scale': reference_scale}
            return inp, out, meta

    def generateSampleFace(self, idx):
        #landmark; shape = (68,2)
        pts = scipy.io.loadmat(os.path.join(self.img_folder, 'landmarks', self.anno[idx].split('_')[0], self.anno[idx][:-4] + '.mat'))['pt3d']
        #image
        sf = self.scale_factor
        rf = self.rot_factor

        imagepath = os.path.join(self.img_folder, self.anno[idx].split('_')[0], self.anno[idx][:-4] + '.jpg')

        img = load_image(imagepath)

        height = img.size(1)
        width = img.size(2)
        hw = max(width, height)
        #consider the landmarks are mainly distributed in the lower half face, so it needs move the center to some lower along height direction
        center = torch.FloatTensor(( float(width*1.0/2), float(height*1.0/2 + height*0.12) ))
        # center = torch.FloatTensor(( float(width*1.0/2), float(height*1.0/2) ))

        reference_scale = torch.tensor(200.0)
        #we hope face for train to be larger than in raw image, so edge will be short by 0.8 ration
        scale_x = hw*0.8 / reference_scale
        scale_y = hw*0.8 / reference_scale
        scale = torch.FloatTensor(( scale_x, scale_y ))

        rotate = 90
        if self.is_train:
            scale[0] = scale[0] * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            scale[1] = scale[0]
            # rotate = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0
            # rotate = 0

            # if random.random() <= 0.5:
            img = torch.from_numpy(fliplr(img.numpy())).float()
            pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
            center[0] = img.size(2) - center[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        img_list = []
        out_list = []
        for i in range(0,37):
            rotate = i * 10
            inp = crop(img, center, scale, reference_scale, [256, 256], rot=rotate)
            r_img = (inp.numpy().transpose(1,2,0)*255).astype(np.uint8)
            img_list.append(cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR))

            tpts = pts.copy()
            out = torch.zeros(self.nParts, 64, 64)
            for i in range(self.nParts):
                if tpts[i, 0] > 0:
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], center, scale, reference_scale, [64, 64], rot=rotate))
                    out[i] = draw_labelmap(out[i], tpts[i], sigma=1)
            out_list.append(out)
        return img_list, out_list, pts, center, scale, reference_scale

def landmark_check(data):
    temp = list(data)
    imgs, heatmaps, pts = temp
    for i in tqdm.tqdm(range(len(imgs))):
        img = imgs[i]
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
        # cv2.imwrite('./rotate/{}.{}.jpg'.format(flag, i), img)


    # return img, pts, pts_img

global flag
if __name__=="__main__":
    # dtset = W300LP(split='train', type='std')
    # global flag
    # for i in range(20):
    #     flag = i
    #     landmark_check(dtset[i])
    from torchsummary import summary
    inp = torch.randn((1,3,192,192))
    summary(model, inp)