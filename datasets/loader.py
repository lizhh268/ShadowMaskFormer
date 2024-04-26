import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


### rotate and flip
# class Augment_RGB_numpy:
#     def __init__(self):
#         pass
#
#     def transform0(self, numpy_array):
#         return numpy_array
#
#     def transform1(self, numpy_array):
#         return np.rot90(numpy_array, k=1, axes=(-2,-1))
#
#     def transform2(self, numpy_array):
#         return np.rot90(numpy_array, k=2, axes=(-2,-1))
#
#     def transform3(self, numpy_array):
#         return np.rot90(numpy_array, k=3, axes=(-2,-1))
#
#     def transform4(self, numpy_array):
#         return np.flip(numpy_array, axis=-2)
#
#     def transform5(self, numpy_array):
#         return np.flip(np.rot90(numpy_array, k=1, axes=(-2,-1)), axis=-2)
#
#     def transform6(self, numpy_array):
#         return np.flip(np.rot90(numpy_array, k=2, axes=(-2,-1)), axis=-2)
#
#     def transform7(self, numpy_array):
#         return np.flip(np.rot90(numpy_array, k=3, axes=(-2,-1)), axis=-2)


### mix two images
# class MixUp_AUG:
#     def __init__(self):
#         self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))
#
#     def aug(self, rgb_gt, rgb_noisy, gray_mask):
#         bs = rgb_gt.size(0)
#         indices = torch.randperm(bs)
#         rgb_gt2 = rgb_gt[indices]
#         rgb_noisy2 = rgb_noisy[indices]
#         gray_mask2 = gray_mask[indices]
#         # gray_contour2 = gray_mask[indices]
#         lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()
#
#         rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
#         rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
#         gray_mask = lam * gray_mask + (1-lam) * gray_mask2
#         return rgb_gt, rgb_noisy, gray_mask



# def align(imgs=[], size=320):
#     H, W, _ = imgs[0].shape
#     Hc, Wc = [size, size]
#
#     Hs = (H - Hc) // 2
#     Ws = (W - Wc) // 2
#     for i in range(len(imgs)):
#         imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
#     return imgs

# def crop_images(imgs, ps=320):
#     _, H, W = imgs.shape
#     r = 0 if H - ps == 0 else np.random.randint(0, H - ps)
#     c = 0 if W - ps == 0 else np.random.randint(0, W - ps)
#     imgs = imgs[:, r:r+ps, c:c+ps]
#     return imgs

# augment  = Augment_RGB_numpy()
# transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'gt')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, 'cond', img_name)) * 2 - 1
        target_img = read_img(os.path.join(self.root_dir, 'gt', img_name)) * 2 - 1
        mask_img = read_img(os.path.join(self.root_dir, 'mask', img_name)) * 2 - 1

        source_img = hwc_to_chw(source_img)
        target_img = hwc_to_chw(target_img)
        mask_img = hwc_to_chw(mask_img)
        #condmask_img = hwc_to_chw(condmask_img)

        # apply_trans = transforms_aug[random.getrandbits(3)]
        # source_img = getattr(augment, apply_trans)(source_img)
        # target_img = getattr(augment, apply_trans)(target_img)
        # mask_img = getattr(augment, apply_trans)(mask_img)
        #[source_img, target_img, mask_img] = [source_img, target_img, mask_img]

        source_img = np.ascontiguousarray(source_img)
        target_img = np.ascontiguousarray(target_img)
        mask_img = np.ascontiguousarray(mask_img)
        #condmask_img = np.ascontiguousarray(condmask_img)
        return {'source': (source_img), 'target': (target_img),'mask':(mask_img), 'filename': img_name}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}