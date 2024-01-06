import mvtec
from mvtec import MVTecDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

import numpy as np
import functools
import os
import random
import argparse
import warnings
import gc
from unet import UNet
from torch_ema import ExponentialMovingAverage

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser('configuration')
    parser.add_argument('--num_iter', type=int, default=3)
    parser.add_argument('--perturbed_t', type=int, default=1e-3)
    parser.add_argument('--dataset_path', type=str, default='./mvtec/')
    parser.add_argument('--pretrained_weights_path', type=str, default='./save/models/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--class_name', type=str, default='all')
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def marginal_prob_std(t, sigma, device):
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device):
    return torch.tensor(sigma**t, device=device)
    
def roc_auc_pxl(gt, score):
    pixel_auroc = roc_auc_score(gt.flatten(), score.flatten())
    return pixel_auroc

def cal_pxl_roc(gt_mask, scores):
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    pixel_auroc = roc_auc_pxl(gt_mask.flatten(), scores.flatten())
    return fpr, tpr, pixel_auroc

def roc_auc_img(gt, score):
    img_auroc = roc_auc_score(gt, score)
    return img_auroc

def cal_img_roc(scores, gt_list):
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_auroc = roc_auc_img(gt_list, img_scores)
    return fpr, tpr, img_auroc

def run():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    class_names = mvtec.CLASS_NAMES if args.class_name == 'all' else [args.class_name]
    
    for class_name in class_names:
        test_dataset   = MVTecDataset(dataset_path  = args.dataset_path, 
                                      class_name    =  class_name, 
                                      is_train      =  False)

        test_loader    = DataLoader(dataset         =   test_dataset, 
                                    batch_size      =   args.batch_size, 
                                    pin_memory      =   True,
                                    shuffle         =   False,
                                    drop_last       =   False,
                                    num_workers     =   args.num_workers)
        
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma = 25, device = device)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma = 25, device = device)
        
        score_model = UNet(marginal_prob_std = marginal_prob_std_fn,
                           n_channels        = 3,
                           n_classes         = 3,
                           embed_dim         = 256)
        load_pth = f"{args.pretrained_weights_path}/{class_name}.pth"
        ckpt = torch.load(load_pth, map_location=device)
        score_model = score_model.to(device)
        score_model.load_state_dict(ckpt)
        
        all_scores = None
        all_mask = None
        all_x = None
        all_y = None

        for x, y, mask in test_loader:
            x = x.to(device)
            sample_batch_size = x.shape[0]
            t = torch.ones(sample_batch_size, device=device) * args.perturbed_t

            scores = 0.
            with torch.no_grad():
                for i in range(args.num_iter):
                    ms = marginal_prob_std_fn(t)[:, None, None, None]
                    g = diffusion_coeff_fn(t)[:, None, None, None]
                    n = torch.randn_like(x)*ms
                    z = x + n
                    score = score_model(z, t)
                    score = score*ms**2 + n
                    scores += (score**2).mean(1, keepdim = True)
            scores /= args.num_iter

            all_scores = torch.cat((all_scores, scores), dim = 0) if all_scores != None else scores
            all_mask = torch.cat((all_mask,mask), dim = 0) if all_mask != None else mask
            all_x = torch.cat((all_x,x), dim = 0) if all_x != None else x
            all_y = torch.cat((all_y,y), dim = 0) if all_y != None else y

        heatmaps = all_scores.cpu().detach().sum(1, keepdim = True)
        heatmaps = F.interpolate(torch.Tensor(heatmaps), (256, 256), mode = "bilinear", align_corners=False)
        heatmaps = F.avg_pool2d(heatmaps, 31,1, padding = 15).numpy()
        
        _, _, img_auroc = cal_img_roc(heatmaps.max(axis = (1,2,3)), all_y)
        _, _, pixel_auroc = cal_pxl_roc(all_mask, heatmaps)

        print('Category : {}, img_auroc : {}, pixel_auroc : {}'.format(class_name, round(img_auroc,3), round(pixel_auroc,3)))

if __name__ == '__main__':
    setup_seed(7777)
    run()