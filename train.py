import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from skimage.metrics import structural_similarity
from utils import AverageMeter,chw_to_hwc
from datasets.loader import PairLoader
#from datasets.loader import MixUp_AUG
from models import *
import cv2
import random
import numpy as np
from torchprofile import profile_macs
from skimage import color
from collections import OrderedDict
#from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='shadowmaskformer-b', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='E:/ISTD/data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='ISTD', type=str, help='dataset name')
parser.add_argument('--exp', default='istd', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler, epoch):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        mask_img = batch['mask'].cuda()
        #nonmask_img = batch['nonmask'].cuda()
        #condmask_img = batch['condmask'].cuda()

        # if epoch > 70:
        #     target_img, input_, mask_img = MixUp_AUG().aug(target_img, source_img, mask_img)

        with autocast(args.no_autocast):
            output = network(source_img, mask_img)
            loss = criterion(output, target_img)
        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():							# torch.no_grad() may cause warning
            output = network(source_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':
    calculate = True
    setting_filename = os.path.join('configs', args.exp, args.model+'.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min= 1e-6)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    #dataset_dir = args.data_dir
    train_dataset = PairLoader(dataset_dir, 'train', 'train', setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'valid', setting['valid_mode'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
        best_psnr = 0

        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler, epoch)

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                writer.add_scalar('best_psnr', best_psnr, epoch)

    else:
        print('==> Existing trained model')
        exit(1)
