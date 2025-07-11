# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import os
import argparse
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from models import model_dict, SUPPORTED_MODELS, get_model_info
from datasets import get_chestxray14_dataloaders


def parse_option():

    parser = argparse.ArgumentParser('PyTorch Knowledge Distillation - Teacher training for ChestX-ray14')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='optimizer choice')
    parser.add_argument('--loss', type=str, default='asl', choices=['bce', 'asl', 'focal'], help='loss function choice')

    # Model
    parser.add_argument('--model', type=str, default='vit_timm_teacher', 
                       choices=['vit_teacher', 'vit_timm_teacher', 'vit_small'])
    parser.add_argument('--timm_model', type=str, default='vit_base_patch16_224', 
                       choices=['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_huge_patch14_224', 
                               'deit_base_patch16_224', 'deit_large_patch16_224'],
                       help='timm model name for pretrained ViT')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='chestxray14', help='dataset')

    # Tham số cho ViT
    parser.add_argument('--patch_size', type=int, default=16, help='kích thước patch cho ViT')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # Warmup settings
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--warmup_method', type=str, default='linear', choices=['linear', 'cosine'], help='warmup method')

    opt = parser.parse_args()

    # Validate timm model name
    if opt.model == 'vit_timm_teacher' and opt.timm_model not in SUPPORTED_MODELS:
        print(f"Error: {opt.timm_model} is not supported.")
        print(f"Supported models: {SUPPORTED_MODELS}")
        exit(1)

    # set the path according to the environment
    opt.model_path = './save/teacher/models'
    opt.tb_path = './save/teacher/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_trial_{}'.format(opt.dataset, opt.model, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    patience = 20  # Tăng từ 10 lên 20 epochs: số epoch không cải thiện liên tiếp cho phép
    counter = 0    # Đếm số epoch không cải thiện
    best_auc = 0

    opt = parse_option()

    #=====================data=====================
    train_loader, val_loader = get_chestxray14_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    n_cls = 14  # ChestX-ray14 có 14 classes

    #=====================model=====================
    if opt.model == 'vit_timm_teacher':
        # Sử dụng timm pretrained model
        model = model_dict[opt.model](num_classes=n_cls, pretrained=True, model_name=opt.timm_model)
        print(f"Using timm pretrained model: {opt.timm_model}")
        
        # Hiển thị thông tin chi tiết về model
        model_info = get_model_info(opt.timm_model)
        if model_info:
            print(f"Model info: {model_info['params']} parameters, "
                  f"embed_dim={model_info['embed_dim']}, "
                  f"depth={model_info['depth']}, "
                  f"heads={model_info['heads']}")
    elif opt.model == 'vit_small':
        # Sử dụng vit_small làm baseline (không pretrained)
        model = model_dict[opt.model](num_classes=n_cls, pretrained=False)
        print(f"Using vit_small as baseline model (no pretrained weights)")
    else:
        # Sử dụng custom ViT teacher
        model = model_dict[opt.model](num_classes=n_cls, patch_size=opt.patch_size)
        print(f"Using custom ViT model with patch_size={opt.patch_size}")
    
    # Tính số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    #=====================optimizer=====================
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    #=====================loss function=====================
    if opt.loss == 'bce':
        # Sử dụng BCEWithLogitsLoss với pos_weight cho imbalanced data
        pos_weight = torch.ones(n_cls) * 2.0  # Tăng weight cho positive samples
        if torch.cuda.is_available():
            pos_weight = pos_weight.cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif opt.loss == 'asl':
        # Sử dụng Asymmetric Loss (ASL) - mặc định
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    else:  # focal
        # Focal Loss implementation
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
                return focal_loss.mean()
        criterion = FocalLoss(alpha=1, gamma=2)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    #=====================tensorboard=====================
    # logger = SummaryWriter(opt.tb_folder)

    #=====================routine=====================
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate_with_warmup(epoch, opt, optimizer)
        print("==> Training...")

        time1 = time.time()
        train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('Epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.add_scalar('train_acc', train_acc, epoch)
        # logger.add_scalar('train_loss', train_loss, epoch)

        val_auc, val_loss = validate(val_loader, model, criterion, opt)

        # logger.add_scalar('val_auc', val_auc, epoch)
        # logger.add_scalar('val_loss', val_loss, epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0  # Reset nếu có cải thiện
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'val_auc': val_auc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    print('==> Best Avg. AUC:', best_auc)

   #=====================save=====================
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)
    # logger.close()


def train(epoch, train_loader, model, criterion, optimizer, opt):
    """ Vanilla training """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for idx, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        # Tính accuracy cho multi-label classification
        losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, _) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            all_outputs.append(output)
            all_targets.append(target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses))

        # Sau khi duyệt hết val_loader
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        aucs, avg_auc = compute_auc_multilabel(all_outputs, all_targets)
        print(' * Avg. AUC {:.4f}'.format(avg_auc))
        print(' * AUC per class:', ['{:.4f}'.format(a) for a in aucs])

    return avg_auc, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_multilabel(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for multi-label classification"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        
        # For multi-label, we consider a prediction correct if any of the predicted labels match
        # This is a simplified approach - you might want to use different metrics
        pred = (output > 0.5).float()
        correct = (pred == target).float().sum(dim=1)
        correct = (correct > 0).float()  # Consider correct if at least one label is correct
        
        res = []
        for k in topk:
            correct_k = correct.sum().item()
            res.append(correct_k * 100.0 / batch_size)
        return res


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    for param_group in optimizer.param_groups:
        if (epoch + 1) in LUT:
            param_group['lr'] = LUT[epoch + 1]


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if opt.lr_decay_epochs is None:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.learning_rate
        for decay_epoch in opt.lr_decay_epochs:
            if epoch >= decay_epoch:
                param_group['lr'] *= opt.lr_decay_rate
        print('learning rate:', param_group['lr'])


def adjust_learning_rate_with_warmup(epoch, opt, optimizer):
    """Learning rate scheduler with warmup"""
    if epoch <= opt.warmup_epochs:
        # Warmup phase
        if opt.warmup_method == 'linear':
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:  # cosine
            lr = opt.learning_rate * 0.5 * (1 + np.cos(np.pi * (1 - epoch / opt.warmup_epochs)))
    else:
        # Normal decay phase
        lr = opt.learning_rate
        for decay_epoch in opt.lr_decay_epochs:
            if epoch >= decay_epoch:
                lr *= opt.lr_decay_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate:', lr)


def compute_auc_multilabel(output, target):
    """
    output: tensor (batch_size, n_classes), logits
    target: tensor (batch_size, n_classes), 0/1
    Returns: (list_auc, avg_auc)
    """
    output = torch.sigmoid(output).detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    n_classes = target.shape[1]
    aucs = []
    for i in range(n_classes):
        try:
            auc = roc_auc_score(target[:, i], output[:, i])
        except ValueError:
            auc = float('nan')  # Nếu chỉ có 1 class trong batch
        aucs.append(auc)
    avg_auc = np.nanmean(aucs)
    return aucs, avg_auc


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


if __name__ == '__main__':
    main()