import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='shadowmaskformer-b', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='ISTD', type=str, help='dataset name')
parser.add_argument('--exp', default='istd', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        mask = batch['mask'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input, mask).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)


if __name__ == '__main__':
    network = eval(args.model.replace('-', '_'))()
    network.cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)