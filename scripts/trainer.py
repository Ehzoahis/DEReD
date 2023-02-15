import sys
sys.path.append('..')

import numpy as np
from tqdm.auto import tqdm, trange
import glob
import wandb

import torch
import torch.nn as nn
import os
import pandas as pd

from model import *
from utils import *

class Trainer():
    def __init__(self, dataloaders, model, render, camera, optimizer, scheduler, args):
        self.train_dl = dataloaders[0]
        self.valid_dl = dataloaders[1]
        self.test_dl = dataloaders[2]
        self.model = model 
        self.render = render
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
        self.camera = camera

        self.args = args
        self.n_iter = 0
        self.iterperEpoch = len(self.train_dl)

        self.save_path = os.path.join('../exp/', args.name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model_savepath = os.path.join(self.save_path, 'model')
        if not os.path.exists(self.model_savepath):
            os.mkdir(self.model_savepath)
        self.eval_savepath = os.path.join(self.save_path, 'result')
        if not os.path.exists(self.eval_savepath):
            os.mkdir(self.eval_savepath)

    def train(self):
        early_stop = 0
        min_val_loss = np.inf
        if self.args.verbose:
            iter_ = trange(self.n_iter//self.iterperEpoch, self.args.epoch, dynamic_ncols=True, unit='Epoch')
        else:
            iter_ = range(self.n_iter//self.iterperEpoch, self.args.epoch)

        for e in iter_:
            self.model.train()
            train_loss = self._run_one_epoch()

            if self.args.save_checkpoint and e % self.args.SAVE_FREQ == 0:
                state = {
                    'iter': self.n_iter,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(state, os.path.join(self.model_savepath, f'checkpoint-{e}.pth'))

            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_one_epoch(cross_valid=True)
                if min_val_loss > val_loss:
                    early_stop = 0
                    if self.args.save_best:
                        state = {
                        'iter': self.n_iter,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }
                        torch.save(state, os.path.join(self.model_savepath, f'best-model.pth'))
                    min_val_loss = val_loss
                else:
                    early_stop += 1
                    if self.args.early_stop and early_stop > self.args.early_stop:
                        print('Loss no longer decrease, Stop...')
                        break
        
                if self.args.vis and e % self.args.VIS_FREQ == 0:
                    self.visualize()
        
        if self.args.save_last:
            state = {
                'model': self.model.state_dict(),
                'iter': self.n_iter,
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(state, os.path.join(self.model_savepath, f'final-model.pth'))

    def _run_one_epoch(self, cross_valid=False):
        total_loss = 0
        dataloader = self.train_dl if not cross_valid else self.valid_dl

        for i, batch in enumerate(dataloader):
            if self.args.use_cuda:
                for key, val in batch.items():
                    batch[key] = val.cuda() 
            
            B, FS, C, H, W = batch['output'].shape
            aif_d = self.model(batch['rgb_fd'])
            fds = batch['output_fd']
            
            aif = torch.clip(aif_d[:, :-1], 0, 1)
            gt_aif = batch['aif']
            dpt = batch['dpt'] if self.args.gt_dpt else aif_d[:, -1]
            dpt = dpt_post_op(dpt, self.args)

            fs_aif = aif.unsqueeze(1).expand(B, FS, C, H, W).contiguous().view(B*FS, C, H, W)
            fs_dpt = dpt.unsqueeze(1).expand(B, FS, H, W).contiguous().view(B*FS, H, W)
            fd = fds.view(-1, 1, 1).expand_as(fs_dpt)
            defocus = self.camera.getCoC(fs_dpt, fd)
            recon = self.render(fs_aif, defocus)

            comp_fs = batch['output'] if self.args.recon_all else batch['rgb_fd'][:, :, :-1]
            clear_idx = torch.argmin(defocus.view(B, FS, 1, H, W).expand(B, FS, C, H, W), dim=1, keepdim=True)
            coarse_aif = torch.gather(comp_fs, 1, clear_idx).squeeze(1)

            loss = self.cal_loss(aif, dpt, coarse_aif, recon, batch['output'], cross_valid, gt_aif)
            total_loss += loss

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.n_iter += 1

        avg_loss = total_loss / (i + 1)
        if self.args.log and cross_valid:
            logs = dict(loss=loss)
            wandb.log({'test':logs})
        return avg_loss

    def cal_loss(self, aif, dpt, coarse_aif, recon, tar, cross_valid, gt_aif):
        B, FS, C, H, W = tar.shape
        loss_ssim = 1 - self.criterion['ssim'](recon, tar.view(B*FS, 3, H, W))
        loss_l1 = self.criterion['l1'](recon, tar.view(B*FS, 3, H, W))
        loss_sharp = self.criterion['sharp'](recon, tar.view(B*FS, 3, H, W))
        loss_recon = self.args.recon_loss_alpha * loss_ssim + (1 - self.args.recon_loss_alpha) * loss_l1
        defocus_loss = loss_recon * self.args.recon_loss_lambda + loss_sharp * self.args.sharp_loss_lambda 
        loss = defocus_loss
        logs = dict(defocus_ssim=1-loss_ssim.item(), defocus_l1=loss_l1.item(), defocus_recon=loss_recon.item(), sharp=loss_sharp.item())

        if not self.args.gt_aif:
            grey_aif = torch.mean(aif, dim=-3, keepdim=True)
            loss_aif_ssim = 1 - self.criterion['ssim'](aif, coarse_aif.clone().detach())
            loss_aif_l1 = self.criterion['l1'](aif, coarse_aif.clone().detach())
            # loss_aif_ssim = 1 - self.criterion['ssim'](aif, coarse_aif)
            # loss_aif_l1 = self.criterion['l1'](aif, coarse_aif)
            loss_aif_recon = self.args.aif_recon_loss_alpha * loss_aif_ssim + (1 - self.args.aif_recon_loss_alpha) * loss_aif_l1
            loss_aif_blur = self.criterion['blur'](grey_aif)
            aif_loss = self.args.aif_recon_loss_lambda * loss_aif_recon + self.args.aif_blur_loss_lambda * loss_aif_blur
            loss = loss + aif_loss
            logs.update(aif_ssim=1-loss_aif_ssim.item(), aif_l1=loss_aif_l1.item(), aif_recon=loss_aif_recon.item(), aif_blur=loss_aif_blur.item())
        else:
            grey_aif = torch.mean(aif, dim=-3, keepdim=True)
            loss_aif_ssim = 1 - self.criterion['ssim'](aif, gt_aif)
            loss_aif_l1 = self.criterion['l1'](aif, gt_aif)
            loss_aif_recon = self.args.aif_recon_loss_alpha * loss_aif_ssim + (1 - self.args.aif_recon_loss_alpha) * loss_aif_l1
            loss_aif_blur = self.criterion['blur'](grey_aif)
            aif_loss = self.args.aif_recon_loss_lambda * loss_aif_recon + self.args.aif_blur_loss_lambda * loss_aif_blur
            loss = loss + aif_loss
            logs.update(aif_ssim=1-loss_aif_ssim.item(), aif_l1=loss_aif_l1.item(), aif_recon=loss_aif_recon.item(), aif_blur=loss_aif_blur.item())

        if not self.args.gt_dpt:
            grey_coarse_aif = torch.mean(coarse_aif, dim=-3, keepdim=True)
            loss_blur = self.criterion['blur'](grey_coarse_aif)
            loss_smooth = self.criterion['smooth'](dpt, aif) 
            dpt_loss = loss_smooth * self.args.sm_loss_lambda + loss_blur * self.args.blur_loss_lambda
            loss = loss + dpt_loss
            logs.update(smooth=loss_smooth.item(), dpt_blur=loss_blur.item())

        if self.args.log and not cross_valid:
            logs.update(loss=loss)
            wandb.log({'train':logs})

        return loss

    def visualize(self, vis_idx=None):
        test_dataset = self.valid_dl.dataset
        n_data = len(test_dataset)
        if not vis_idx:
            vis_idx = np.random.randint(0, n_data)
        vis_data = test_dataset[vis_idx]
        
        FS, C, H, W = vis_data['output'].shape
        aif_d =  self.model(vis_data['rgb_fd'].unsqueeze(0).cuda())
        fds = vis_data['output_fd']
        
        aif = torch.clip(aif_d[:, :-1], 0, 1)
        dpt = vis_data['dpt'].unsqueeze(0).cuda() if self.args.gt_dpt else aif_d[:, -1]
        dpt = dpt_post_op(dpt, self.args)

        fs_aif = aif.expand(FS, C, H, W).contiguous()
        fs_dpt = dpt.expand(FS, H, W).contiguous()
        fd = fds.view(-1, 1, 1).expand_as(fs_dpt).cuda()
        defocus = self.camera.getCoC(fs_dpt, fd)
        recon = self.render(fs_aif, defocus).detach().cpu().numpy() # FS C H W

        defocus_gt = []
        defocus_recon = []
        for i in range(FS):
            # collect reconn defocus
            recon_def = recon[i].transpose(1, 2, 0)
            recon_wandb = wandb.Image(recon_def, caption=f"Recon Defocus, fd={fds[i]}")
            defocus_recon.append(recon_wandb)
            # collect gt defocus
            gt_def = vis_data['output'][i].numpy().transpose(1, 2, 0)
            gt_wandb = wandb.Image(gt_def, caption=f"GT Defocus, fd={vis_data['output_fd'][i]}")
            defocus_gt.append(gt_wandb)
        logs = dict(defocus_gt = defocus_gt, defocus_recon=defocus_recon)
        
        if not self.args.gt_dpt:
            recon_dpt = dpt[0].unsqueeze(-1).detach().cpu().numpy()
            recon_dpt_wandb = wandb.Image(recon_dpt / self.args.camera_far, caption='Recon Dpt')
            try:
                gt_dpt = vis_data['dpt'][0].unsqueeze(-1).squeeze().numpy()
                gt_dpt_wandb = wandb.Image(gt_dpt / self.args.camera_far, caption='GT Dpt')
            except:
                gt_dpt_wandb = recon_dpt_wandb
            logs.update(gt_dpt=gt_dpt_wandb, recon_dpt=recon_dpt_wandb)

        recon_aif = aif[0].detach().cpu().numpy().transpose(1, 2, 0)
        recon_aif_wandb = wandb.Image(recon_aif, caption='Recon Aif')
        try:
            gt_aif = vis_data['aif'].numpy().transpose(1, 2, 0)
            gt_aif_wandb = wandb.Image(gt_aif, caption='GT Aif')
        except:
            gt_aif_wandb = recon_aif_wandb
        logs.update(gt_aif=gt_aif_wandb, recon_aif=recon_aif_wandb)

        wandb.log(logs)

    def load_checkpoint(self, model_name, orig_model=None):
        if not orig_model:
            model_path = os.path.join(self.model_savepath, model_name)
        else:
            model_path = os.path.join('../exp/', orig_model)
        model_dict = torch.load(model_path)
        state_dict = model_dict['model']
        optim_dict = model_dict['optimizer']
        self.model.load_state_dict(state_dict)
        self.n_iter = model_dict['iter'] if not orig_model else 0
        self.optimizer.load_state_dict(optim_dict)
        if not orig_model:
            print(f'Loading Model from Epoch {self.n_iter//self.iterperEpoch}')
        else:
            print(f'Loading Model from {orig_model}')

    def eval_model(self, far=None):
        dataloader = self.test_dl
        if self.args.save_best:
            print('Evaluating the Best Model...')
            model_path = os.path.join(self.model_savepath, 'best-model.pth')
            model_dict = torch.load(model_path)
            state_dict = model_dict['model']
            self.model.load_state_dict(state_dict)
            n_iter = model_dict['iter']
        else:
            print('Evaluating the Lastest Model...')
        df_model = dict()
        if not self.args.gt_dpt:
            df_model.update(Setting=[], AbsRel=[], SqRel=[], RMSE=[], RMSE_log=[], delta1=[], delta2=[], delta3=[])
            if self.args.scale != 1:
                upsample = nn.Upsample(scale_factor=self.args.scale, mode='bilinear', align_corners=True)
        if not self.args.gt_aif:
            df_model.update(MG=[], SF=[])
        for i, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                raw_aif_dpt = self.model(batch['rgb_fd'].cuda())
                raw_aif = raw_aif_dpt[:, :-1]
                raw_dpt = raw_aif_dpt[:, -1]
            if not self.args.gt_dpt:
                pred_dpt = dpt_post_op(raw_dpt, self.args)
                if self.args.scale != 1:
                    pred_dpt = upsample(pred_dpt.unsqueeze(0)).squeeze(0)
                pred_dpt = pred_dpt.cpu()
                dpt_gt = batch['dpt']
                if far is not None:
                    msk = dpt_gt < far
                else:
                    msk = None
                AbsRel, SqRel, RMSE, RMSE_log, delta1, delta2, delta3 = eval_depth(pred_dpt, dpt_gt, msk=msk)
                df_model['AbsRel'].append(AbsRel.numpy())
                df_model['SqRel'].append(SqRel.numpy())
                df_model['RMSE'].append(RMSE.numpy())
                df_model['RMSE_log'].append(RMSE_log.numpy())
                df_model['delta1'].append(delta1.numpy())
                df_model['delta2'].append(delta2.numpy())
                df_model['delta3'].append(delta3.numpy())
            if not self.args.gt_aif:
                pred_aif = torch.clip(raw_aif, 0, 1)
                MG, SF = eval_aif(pred_aif.cpu())
                df_model['MG'].append(MG.numpy())
                df_model['SF'].append(SF.numpy())
            df_model['Setting'].append(self.args.name)
        df_model = pd.DataFrame.from_dict(df_model)
        df_model.to_csv(os.path.join(self.eval_savepath, f'eval{far}.csv'))
