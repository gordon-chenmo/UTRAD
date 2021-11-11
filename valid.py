import argparse
import os
import numpy as np
import time
import datetime
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from torchvision.utils import save_image
from torchvision import models
from PIL import Image

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = TrainOptions().parse()
torch.manual_seed(args.seed)

save_dir = '%s-%s/%s/%s' % (args.exp_name, args.dataset_name, args.model_result_dir, 'checkpoint.pth')
start_epoch = 0
transformer = Create_nets(args)
transformer = transformer.to(device)
transformer.cuda()
'''
if os.path.exists('resnet18_pretrained.pth'):
    backbone = models.resnet18(pretrained=False).to(device)
    backbone.load_state_dict(torch.load('resnet18_pretrained.pth'))
else:
    backbone = models.resnet18(pretrained=True).to(device)
    '''

backbone = models.resnet18(pretrained=True).to(device)

if os.path.exists(save_dir):
    checkpoint = torch.load(save_dir)
    transformer.load_state_dict(checkpoint['transformer'])
    start_epoch = checkpoint['start_epoch']

backbone.eval()
outputs = []
def hook(module, input, output):
    outputs.append(output)
backbone.layer1[-1].register_forward_hook(hook)
backbone.layer2[-1].register_forward_hook(hook)
backbone.layer3[-1].register_forward_hook(hook)
#backbone.layer4[-1].register_forward_hook(hook)
layer = 3

train_dataloader, test_dataloader = Get_dataloader(args)


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


img_save_dir='%s-%s/%s' % (args.exp_name, args.dataset_name, args.validation_image_dir)
if not os.path.exists(img_save_dir):
    os.mkdir(img_save_dir)
score_map = []
gt_list = []
gt_mask_list = []
if True:
    transformer.eval()
    for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader):                       
        with torch.no_grad():
            
            num = 4
            norm_range = [(0.9,1.3),(0.9,1.3),(0.9,1.3),(0.9,1.3),(1.1,1.5),]

            inputs = batch.to(device)
            ground_truth = ground_truth.to(device)
            outputs = []
            _ = backbone(inputs)
            outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
            recon, std = transformer(outputs)
            batch_size, channels, width, height = recon.size()

            dist = torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).div(std.abs())
            dist = dist.view(batch_size, 1, width, height)

            patch_normed_score = []
            for j in range(4):
                patch_size = pow(4, j)
                patch_score = F.conv2d(input=dist, 
                    weight=(torch.ones(1,1,patch_size,patch_size) / (patch_size*patch_size)).to(device), 
                    bias=None, stride=patch_size, padding=0, dilation=1)
                patch_score = F.avg_pool2d(dist,patch_size,patch_size)
                patch_score = F.interpolate(patch_score, (width,height), mode='bilinear')
                patch_normed_score.append(patch_score)
            score = torch.zeros(batch_size,1,64,64).to(device)
            for j in range(4):
                score = embedding_concat(score, patch_normed_score[j])
            
            score = F.conv2d(input=score, 
                    weight=torch.tensor([[[[0.0]],[[0.25]],[[0.25]],[[0.25]],[[0.25]]]]).to(device), 
                    bias=None, stride=1, padding=0, dilation=1)

            score = F.interpolate(score, (ground_truth.size(2),ground_truth.size(3)), mode='bilinear')
            heatmap = score.repeat(1,3,1,1)
            score_map.append(score.cpu())
            gt_mask_list.append(ground_truth.cpu())
            gt_list.append(gt)

            save_image(inputs, '%s-%s/%s/%s' % (args.exp_name, args.dataset_name, 'validation_images',str(i)+'test1_inputs.png'),nrow= num)
            save_image(heatmap, '%s-%s/%s/%s' % (args.exp_name, args.dataset_name, 'validation_images',str(i)+'test3_heatmap.png'),range=norm_range[4],normalize=True,nrow= num)
            save_image(ground_truth, '%s-%s/%s/%s' % (args.exp_name, args.dataset_name, 'validation_images',str(i)+'test2_truth.png'),normalize=True,nrow= num)
            cv2.waitKey(100)
            heatmap2 = cv2.imread('%s-%s/%s/%s' % (args.exp_name, args.dataset_name, 'validation_images',str(i)+'test3_heatmap.png'),cv2.IMREAD_GRAYSCALE)
            heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
            cv2.imwrite('%s-%s/%s/%s' % (args.exp_name, args.dataset_name, 'validation_images',str(i)+'test5_heatmap_color.png'),heatmap2)
            '''
            num = 4
            mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            std = torch.Tensor([0.229, 0.224, 0.225 ]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            batch = batch * std + mean
            save_image(batch, '%s-%s/%s/%s' % (args.exp_name, args.dataset_name, 'validation_images',str(i)+'test1_inputs.png'),
            range=(0,1),normalize=True,nrow= num)
            print('batch',i)'''
#assert 0

score_map = torch.cat(score_map,dim=0)
gt_mask_list = torch.cat(gt_mask_list,dim=0)
gt_list = torch.cat(gt_list,dim=0)
print('dataset: ', args.dataset_name)
if True:
    if True:
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        
        # calculate image-level ROC AUC score
        img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
        gt_list = gt_list.numpy()
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        
        # get optimal threshold
        gt_mask = gt_mask_list.numpy().astype('int')
        scores = scores.numpy().astype('float32')
        '''
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        '''
        '''
        from sklearn.utils.multiclass import type_of_target
        print(type_of_target(gt_mask))
        print(type_of_target(scores))
        '''
        # calculate per-pixel level ROCAUC
        fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten()) 
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten()) 
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
        

        if args.unalign_test:
            with open("./%s-%s/validation_result.log" % (args.exp_name,  args.dataset_name) ,"a") as log:
                log.write('epochs:%ss\n' % (str(start_epoch)))
                log.write('class_name:%s\n' % (args.dataset_name))
                log.write('unalign image ROCAUC: %.3f\n' % (img_roc_auc))
                log.write('unalign pixel ROCAUC: %.3f\n\n' % (per_pixel_rocauc))
        else:
            with open("./%s-%s/validation_result.log" % (args.exp_name,  args.dataset_name) ,"w") as log:
                log.write('epochs:%ss\n' % (str(start_epoch)))
                log.write('class_name:%s\n' % (args.dataset_name))
                log.write('image ROCAUC: %.3f\n' % (img_roc_auc))
                log.write('pixel ROCAUC: %.3f\n\n' % (per_pixel_rocauc))
        #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (args.dataset_name, per_pixel_rocauc))




