from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
import datetime
import glob
import logging
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchvision.datasets as datasets
import yaml
from clearml import Task
from torchsummary import summary

import models
import opts
from datasets import W300LP, landmark_check_image, tensor_to_img
from utils import *
from utils.evaluation import (AverageMeter, accuracy, calc_dists, calc_metrics,
                              final_preds)
from utils.imutils import batch_with_heatmap
from utils.logger import Logger, savefig
from utils.misc import adjust_learning_rate, save_checkpoint, save_pred, show

task = Task.init(project_name="Face-alignment", task_name="Face-landmarks-FAN-MEDIUM-W300LP")

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

BEST_ACC = 0.
BEST_AUC = 0.
IDX = range(1, 69, 1)
EPOCH_FLAG = 0
CHECKPOINT_FLAG = 0
ITER = 0
TODAY = datetime.date.today().strftime("%b-%d-%Y")
NET = ''

args = opts.argparser()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(f'./samples', exist_ok=True) # visualization during validation phase

now = str(datetime.datetime.now())
ms_idx = now.find('.')
now = now[:ms_idx].replace(' ', '_')


def get_loader(data):
    return {
        '300WLP': W300LP,
    }['300WLP']


def load_config(args):
    global NET
    with open(args.config_path, "r") as f:
        model_config = yaml.load(f)

    net_type = args.netType
    model_config = model_config[net_type][args.model_size]
    NET = f"{net_type}-{args.model_size}"
    logging.info(f"=> Using {NET}")
    return net_type, model_config


def train_3dlandmark(args):
    global EPOCH_FLAG
    global BEST_ACC
    global BEST_AUC
    global CHECKPOINT_FLAG
    CHECKPOINT_FLAG = args.checkpoint

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    logging.info("=> Models will be saved at: {}".format(args.checkpoint))

    # nStack=4, nModules=1, nHgDepth=4, num_feats=256, num_classes=68
    netType, model_config = load_config(args)
    logging.info("==> Creating model '{}-{}', stacks={} modules={}, hgdepths={} feats={} classes={}".format(
        args.netType, args.pointType, model_config['nStack'], model_config['nModules'], model_config['nHgDepth'], model_config['nFeats'], model_config['nClasses']))

    if args.netType == 'HourglassNet':
        model = models.__dict__[netType](
            nStack = model_config['nStack'],
            nModules=model_config['nModules'],
            nHgDepth=model_config['nHgDepth'],
            nFeats=model_config['nFeats'],
            nClasses=model_config['nClasses'])
    elif args.netType.startswith('mobilenet'):
        model = models.__dict__[netType](
            num_classes=model_config['nClasses'])
    elif args.netType == 'FAN':
        model = models.__dict__[netType](
            nStack=model_config['nStack'],
            nModules=model_config['nModules'],
            nHgDepth=model_config['nHgDepth'],
            nFeats=model_config['nFeats'],
            nClasses=model_config['nClasses'],
            InvRes=model_config['InvRes'])
    else:
        raise Exception("Unsupported net type <Techainer>")
    
    if args.pretrained:
        model = torch.load(args.pretrained).to(device)
    else:
        model = model.to(device)

    with torch.no_grad():
        inp = torch.randn(1,3,256,256)
        summary(model, inp)

    criterion = torch.nn.MSELoss(size_average=True).to(device)

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    title = args.checkpoint.split('/')[-1] + ' on ' + args.data.split('/')[-1]

    Loader = get_loader(args.data)

    val_loader = torch.utils.data.DataLoader(
        Loader(data_path=args.data, model_config=model_config, split='val'),
        batch_size=args.val_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            BEST_ACC = checkpoint['BEST_ACC']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])

    cudnn.benchmark = True
    logging.info('=> Total params: %.2fM' % (sum(p.numel() for p in model.parameters())))

    if args.evaluation:
        logging.info('=> Evaluation only')
        D = args.data.split('/')[-1]
        save_dir = os.path.join(args.checkpoint, D)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        loss, acc, predictions, auc = validate(val_loader, model, criterion, args.netType,
                                                        args.debug, args.flip)
        save_pred(predictions, checkpoint=save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Loader(data_path=args.data, model_config=model_config, split='train'),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        EPOCH_FLAG = epoch

        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        logging.info("=> Training phase")
        logging.info('==> Epoch: %d | LR %.8f' % (epoch + 1, lr))

        train(train_loader, model, criterion, optimizer, args.netType, args.debug, args.flip)
        validate(val_loader, model, criterion, args.netType, args.debug, args.flip)


def train(loader, model, criterion, optimizer, netType, debug=False, flip=False):
    global ITER

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    gt_win, pred_win = None, None

    for i, (imgs, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        imgs_var = torch.autograd.Variable(imgs.to(device))
        targets_var = torch.autograd.Variable(targets.to(device))

        if debug:
            gt_batch_img = batch_with_heatmap(imgs, targets)
            # pred_batch_img = batch_with_heatmap(imgs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                # plt.subplot(122)
                # pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                # pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        output = model(imgs_var)
        
        if not isinstance(output, list):
            output = [output]

        score_map = output[-1].data.cpu()

        if flip:
            flip_imgs_var = torch.autograd.Variable(
                torch.from_numpy(shufflelr(imgs.clone().numpy())).float().to(device))
            flip_output_var = model(flip_imgs_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o, targets_var)
        acc, _ = accuracy(score_map, targets.cpu(), IDX, thr=0.07)

        losses.update(loss.item(), imgs.size(0))
        acces.update(acc[0], imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%33==32:
            logging.info(f"===> Current loss: {losses.avg} - Current acc: {acces.avg}")
            writer.add_scalar('Loss/Train', losses.avg, ITER)
            writer.add_scalar('Acc/Train', acces.avg, ITER)
            ITER += 1

    return 

def validate(loader, model, criterion, netType, debug, flip):
    logging.info("=> Validation phase")
    global BEST_ACC

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    magic = np.ceil(len(loader)/10)

    with torch.no_grad():
        model.eval()
        gt_win, pred_win = None, None
        all_dists = torch.zeros((68, loader.dataset.__len__()))
        for i, (imgs, targets, meta) in enumerate(loader):

            imgs_var = torch.autograd.Variable(imgs.to(device))
            targets_var = torch.autograd.Variable(targets.to(device))

            output = model(imgs_var)
            score_map = output[-1].data.cpu()

            if flip:
                flip_imgs_var = torch.autograd.Variable(
                    torch.from_numpy(shufflelr(imgs.clone().numpy())).float().to(device))
                flip_output_var = model(flip_imgs_var)
                flip_output = flip_back(flip_output_var[-1].data.cpu())
                score_map += flip_output

            # intermediate supervision
            loss = 0
            for o in output:
                loss += criterion(o, targets_var)
            acc, batch_dists = accuracy(score_map, targets.cpu(), IDX, thr=0.07)
            all_dists[:, i * args.val_batch:(i + 1) * args.val_batch] = batch_dists

            if i==magic:
                logging.info("==> Saving samples!")
                pts, pts_img = get_preds_fromhm(score_map, meta['center'], meta['scale'], meta['reference_scale'])
                for i in range(len(imgs)):
                    arr_img = tensor_to_img(imgs[i])
                    arr_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
                    for _pts in pts[i]:
                        cv2.circle(arr_img, (_pts[0], _pts[1]),2,(0,255,0), -1, 8)
                    cv2.imwrite(f'./samples/{i}.jpg', arr_img)
            # os.makedirs(f'./samples', exist_ok=True)
            # pts, pts_img = get_preds_fromhm(score_map, meta['center'], meta['scale'], meta['reference_scale'])
            # for i in range(len(imgs)):
            #     arr_img = (imgs[i].numpy().transpose(1,2,0)*255).astype(np.uint8)
            #     arr_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
            #     for _pts in pts[i]:
            #         cv2.circle(arr_img, (_pts[0], _pts[1]),2,(0,255,0), -1, 8)
            #     cv2.imwrite(f'./samples/{i}.jpg', arr_img)

            if debug:
                gt_batch_img = batch_with_heatmap(imgs, targets)
                pred_batch_img = batch_with_heatmap(imgs, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            losses.update(loss.item(), imgs.size(0))
            acces.update(acc[0], imgs.size(0))

        mean_error = torch.mean(all_dists)
        auc = calc_metrics(all_dists) # this is auc of predicted maps and targets.
        writer.add_scalar('AUC/Val', auc, EPOCH_FLAG)
        writer.add_scalar('ME/Val', mean_error, EPOCH_FLAG)
        
        logging.info(f"==> Loss: {losses.avg} - Acc: {acces.avg} - AUC: {auc} - ME: {mean_error}")

        if acces.avg >= BEST_ACC:
            BEST_ACC = acces.avg
            save_path = os.path.join(CHECKPOINT_FLAG, f"best-{BEST_ACC}-{TODAY}.pth")
            torch.save(model, save_path)

            save_path = os.path.join(CHECKPOINT_FLAG, f"last-{acces.avg}-{TODAY}.pth")
            torch.save(model, save_path)
            logging.info("===> Model is saved at {} - current acc is {} - best acc is {}".format(save_path, acces.avg, BEST_ACC))
        else:
            save_path = os.path.join(CHECKPOINT_FLAG, f"last-{acces.avg}-{TODAY}.pth")
            torch.save(model, save_path)
            logging.info("===> Model is saved at {} - current acc is {} - best acc is {}".format(save_path, acces.avg, BEST_ACC))

    return 
    

if __name__ == '__main__':
    #test makefile
    train_3dlandmark(args)
