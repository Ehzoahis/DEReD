import numpy as np
import torch
from .dataset import *
from torch.nn import functional as F

def load_data(dataset_config, dataset, BS, n_shot=-1, indices=None):
    if dataset == 'NYU100':
        train_dl, valid_dl, test_dl = _load_NYU_data(dataset_config, BS, NYU100=True)
    elif dataset == 'DefocusNet':
        train_dl, valid_dl, test_dl = _load_defocus_data(dataset_config, BS)
    elif dataset == 'mobileDFD':
        train_dl, valid_dl, test_dl = _load_mDFD_data(dataset_config, BS, n_shot=n_shot, sel_indices=indices)
    else:
        train_dl, valid_dl, test_dl = _load_NYU_data(dataset_config, BS)
    return train_dl, valid_dl, test_dl

def _load_defocus_data(dataset_config, BS, num_workers=8, valid_split=0.2, dataset_shuffle=True):
    dataset = DefocusNet(**dataset_config)
    indices = np.arange(len(dataset))
    indices_train = indices
    indices_test = indices
    indices_val = indices
    dataset_train = torch.utils.data.Subset(dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(dataset, indices_val)
    dataset_test = torch.utils.data.Subset(dataset, indices_test)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=BS, shuffle=dataset_shuffle, pin_memory=True, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=1, batch_size=1, shuffle=False)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, num_workers=1, batch_size=1, shuffle=False)
    print("Total number of training sample:", len(loader_train.dataset))
    print("Total number of testing sample:", len(loader_valid.dataset))
    print("Total number of testing sample:", len(loader_test.dataset))

    return loader_train, loader_valid, loader_test

def _load_NYU_data(dataset_config, BS, num_workers=8, dataset_shuffle=True, valid_split=0.2, NYU100 = False):
    if NYU100:
        dataset = NYUFS100Dataset
    else:
        dataset = NYUDataset
    
    dataset_train = dataset(**dataset_config, split='train')
    dataset_valid = dataset(**dataset_config, split='test', resize=None)

    indices = np.arange(len(dataset_valid))
    split = int(len(dataset_valid) * (1 - valid_split))
    indices_test = indices
    indices_valid = indices[split:]

    dataset_test = torch.utils.data.Subset(dataset_valid, indices_test)
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=BS, shuffle=dataset_shuffle, pin_memory=True, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=num_workers, batch_size=BS, shuffle=False, pin_memory=True)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

    print("Total number of training sample:", len(dataset_train))
    print("Total number of validation sample:", len(indices_valid))
    print("Total number of testing sample:", len(indices_test))

    return loader_train, loader_valid, loader_test

def _load_mDFD_data(dataset_config, BS, num_workers=8, dataset_shuffle=True, n_shot=-1, sel_indices=None):
    dataset = MobileDFD(**dataset_config)

    indices = np.arange(len(dataset))
    if n_shot != -1:
        if sel_indices is None:
            np.random.shuffle(indices)
            indices = np.sort(indices[:n_shot])
        else:
            indices = sel_indices
            if len(sel_indices) != n_shot:
                print(f"{len(sel_indices)} != {n_shot}, use {len(sel_indices)} shots")
        print(f"Using Indices: {indices}")

    sub_dataset = torch.utils.data.Subset(dataset, indices)

    print("Total number of sample:", len(sub_dataset))

    loader = torch.utils.data.DataLoader(dataset=sub_dataset, shuffle=dataset_shuffle, num_workers=num_workers, batch_size=BS, pin_memory=True)

    return loader, loader, loader

class ThinLenCamera():
    def __init__(self, fnumber=0.5, focal_length=2.9*1e-3, sensor_size=3.1*1e-3, img_size=256, pixel_size=None):
        self.focal_length = focal_length
        self.D = self.focal_length / fnumber
        self.pixel_size = pixel_size
        if not self.pixel_size:
            self.pixel_size = sensor_size / img_size
        
    def getCoC(self, dpt, focus_dist):
        # dpt : BxFS H W
        # focus_dist : BxFS H W
        sensor_dist = focus_dist * self.focal_length / (focus_dist - self.focal_length)
        CoC = self.D * sensor_dist * torch.abs(1/self.focal_length - 1/sensor_dist - 1/(dpt+1e-8))
        sigma = CoC / 2 / self.pixel_size
        return sigma.type(torch.float32)

def dpt_post_op(dpt, args):
    B = dpt.shape[0]
    if args.dpt_post_op == 'clip':
        dpt = torch.clip(dpt, 0, args.camera_far)
    elif args.dpt_post_op == 'norm':
        dpt_local_min = torch.min(dpt.view(B, -1), dim=1)[0].view(B, 1, 1)
        dpt_local_max = torch.max(dpt.view(B, -1), dim=1)[0].view(B, 1, 1)
        if args.normalize_dpt:
            norm_dpt_ = (dpt - dpt_local_min)/(dpt_local_max - dpt_local_min + 1e-8)
        else:
            norm_dpt_ = dpt / (dpt_local_max + 1e-8)
        norm_dpt = norm_dpt_ * (args.camera_far - args.camera_near) + args.camera_near
        dpt = norm_dpt
    return dpt

def eval_depth(pred, gt, msk=None):
    error = torch.abs(gt - pred)
    if msk is None:
        AbsRel = torch.mean(error / gt, dim=[1, 2, 3])
        SqRel = torch.mean(error ** 2 / gt, dim=[1, 2, 3])
        RMSE = torch.sqrt(torch.mean(error ** 2, dim=[1, 2, 3]))
        RMSE_log = torch.sqrt(torch.mean(torch.abs(torch.log10(gt+1e-8) - torch.log10(pred+1e-8)) ** 2, dim=[1, 2, 3]))
        gt_pred = gt/(pred+1e-8)
        pred_gt = pred/(gt+1e-8)
        acc = torch.max(gt_pred, pred_gt)
        delta1 = torch.sum(acc < 1.25, dim=[1, 2, 3])/(acc.shape[-1] * acc.shape[-2])
        delta2 = torch.sum(acc < 1.25**2, dim=[1, 2, 3])/(acc.shape[-1] * acc.shape[-2])    
        delta3 = torch.sum(acc < 1.25**3, dim=[1, 2, 3])/(acc.shape[-1] * acc.shape[-2])
        return AbsRel.mean(), SqRel.mean(), RMSE.mean(), RMSE_log.mean(), delta1.mean(), delta2.mean(), delta3.mean()
    else:
        msk_num = torch.sum(msk)
        AbsRel = torch.sum(error * msk / gt * msk) / msk_num
        SqRel = torch.sum(error ** 2 / gt * msk) / msk_num
        RMSE = torch.sqrt(torch.sum(error ** 2 * msk) / msk_num)
        RMSE_log = torch.sqrt(torch.sum(torch.abs(torch.log10(gt+1e-8) - torch.log10(pred+1e-8)) ** 2 * msk) / msk_num)
        gt_pred = gt/(pred+1e-8)
        pred_gt = pred/(gt+1e-8)
        acc = torch.max(gt_pred, pred_gt)
        delta1 = torch.sum((acc < 1.25) * msk)/msk_num
        delta2 = torch.sum((acc < 1.25**2) * msk)/msk_num    
        delta3 = torch.sum((acc < 1.25**3) * msk)/msk_num

        return AbsRel, SqRel, RMSE, RMSE_log, delta1, delta2, delta3


def eval_aif(inp):
    dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
    dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
    MG = torch.mean(torch.sqrt((dx ** 2 + dy ** 2)/2), dim=[1,2,3]) # Large -> Better
    SF = torch.sqrt(torch.mean(dx ** 2, dim=[1,2,3]) + torch.mean(dy ** 2, dim=[1,2,3])) # Large -> Better
    return MG.mean(), SF.mean()