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
from thop import profile, clever_format

from models import model_dict
from models.util import ConvReg, LinearEmbed, Connector, Translator, Paraphraser
from datasets import get_chestxray14_dataloaders, get_chestxray14_dataloaders_sample
from distillers import DistillKL, HintLoss, Attention, Similarity, Correlation, \
                       VIDLoss, RKDLoss, PKT, ABLoss, FactorTransfer, KDSVD, \
                       FSP, NSTLoss, ITLoss, EGA, CRDLoss, RRDLoss


def parse_option():

    parser = argparse.ArgumentParser('PyTorch Knowledge Distillation - Student training for ChestX-ray14')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,75', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='optimizer choice')
    parser.add_argument('--loss', type=str, default='asl', choices=['bce', 'asl', 'focal'], help='loss function choice')

    # Dataset
    parser.add_argument('--dataset', type=str, default='chestxray14', help='dataset')

    # Model
    parser.add_argument('--model_s', type=str, default='vit_student', choices=['vit_student'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # Distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 
                                                                      'correlation', 'vid', 'kdsvd', 
                                                                      'fsp','rkd', 'pkt', 'abound', 'factor',
                                                                      'nst', 'itrd', 'ega',
                                                                      'crd', 'rrd',])
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t_s', default=0.1, type=float, help='student temperature parameter for softmax')
    parser.add_argument('--nce_t_t', default=0.02, type=float, help='teacher temperature parameter for softmax') 
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax') 
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--memory_type', default='fifo', type=str, choices=['fifo', 'momentum'])

    # Other
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    
    opt = parser.parse_args()

    # Set the path according to the environment
    opt.model_path = './save/student/student_model'
    opt.tb_path = './save/student/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = '{}_{}_S_{}_T_{}_r_{}_a_{}_b_{}_trial_{}'.format(opt.distill.upper(), opt.dataset, opt.model_s, opt.model_t, opt.gamma, opt.alpha,  opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    print("CUDA available:", torch.cuda.is_available())
    print("GPUs: ", torch.cuda.device_count())
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU Name: {gpu_info.name}")
        print(f"Total VRAM: {gpu_info.total_memory / 1e9} GB")
    options_dict = vars(opt)
    for key, value in options_dict.items():
        print(f"{key}: {value}")
    return opt


def get_teacher_name(model_path):
    """ Parse teacher name """
    if 'vit_timm_teacher' in model_path:
        return 'vit_timm_teacher'
    elif 'vit_teacher' in model_path:
        return 'vit_teacher'
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]



# Cho phép numpy scalar
#torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.float64])

def load_teacher(model_path, n_cls):
    """ Load teacher model """
    print('==> Loading teacher model')
    model_t = get_teacher_name(model_path)
    
    if model_t == 'vit_timm_teacher':
        # Đối với vit_timm_teacher, cần xác định model_name từ path
        # Path format: .../chestxray14_vit_timm_teacher_trial_0/vit_timm_teacher_best.pth
        path_parts = model_path.split('/')
        folder_name = path_parts[-2]  # chestxray14_vit_timm_teacher_trial_0
        
        # Mặc định sử dụng deit_base_patch16_224 nếu không thể xác định
        model_name = 'deit_base_patch16_224'
        
        # Có thể thêm logic để xác định model_name từ folder_name nếu cần
        print(f'Loading vit_timm_teacher with model: {model_name}')
        model = model_dict[model_t](num_classes=n_cls, model_name=model_name)
    else:
        model = model_dict[model_t](num_classes=n_cls)
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=False)['model'])
    except:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)['model'])
    print('Teacher model loaded')
    return model


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


def calculate_model_stats(model, input_size=(1, 3, 224, 224)):
    """
    Calculate number of parameters and FLOPs for a model
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        flops_formatted: Formatted FLOPs string
    """
    input_tensor = torch.randn(input_size)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs
    try:
        model.eval()
        # Use profile with proper error handling
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_formatted, _ = clever_format([flops], "%.3f")
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        flops_formatted = "N/A"
    
    return total_params, trainable_params, flops_formatted


def format_params(num_params):
    """
    Format number of parameters to human readable format
    Args:
        num_params: Number of parameters
    Returns:
        Formatted string
    """
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return f"{num_params}"


def main():
    best_auc = 0
    patience = 20
    counter = 0

    opt = parse_option()

    #=====================tensorboard=====================
    # logger = SummaryWriter(log_dir=opt.tb_folder)

    #=====================data=====================
    if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
        train_loader, val_loader, n_data = get_chestxray14_dataloaders_sample(batch_size=opt.batch_size, num_workers=opt.num_workers, k=opt.nce_k, mode=opt.mode)
    else:
        train_loader, val_loader, n_data = get_chestxray14_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
    n_cls = 14  # ChestX-ray14 có 14 classes

    #=====================model=====================
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    #=====================print model statistics=====================
    print("\n" + "="*50)
    print("MODEL STATISTICS")
    print("="*50)
    
    # Calculate and print student model stats
    total_params, trainable_params, flops = calculate_model_stats(model_s)
    print(f"Student Model ({opt.model_s}):")
    print(f"  Total Parameters: {format_params(total_params)} ({total_params:,})")
    print(f"  Trainable Parameters: {format_params(trainable_params)} ({trainable_params:,})")
    print(f"  FLOPs: {flops}")
    
    # Calculate and print teacher model stats
    total_params_t, trainable_params_t, flops_t = calculate_model_stats(model_t)
    print(f"Teacher Model ({opt.model_t}):")
    print(f"  Total Parameters: {format_params(total_params_t)} ({total_params_t:,})")
    print(f"  Trainable Parameters: {format_params(trainable_params_t)} ({trainable_params_t:,})")
    print(f"  FLOPs: {flops_t}")
    
    # Calculate compression ratio
    param_ratio = total_params / total_params_t if total_params_t > 0 else 0
    print(f"\nCompression Ratio:")
    print(f"  Parameters: {param_ratio:.2f}x smaller")
    print("="*50 + "\n")

    #=====================mock data=====================
    data = torch.randn(2, 3, 224, 224)  # Kích thước ảnh cho ChestX-ray14
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    #=====================modules=====================
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    #=====================criteria=====================
    # Loss function for classification
    if opt.loss == 'bce':
        # Sử dụng BCEWithLogitsLoss với pos_weight cho imbalanced data
        pos_weight = torch.ones(n_cls) * 2.0  # Tăng weight cho positive samples
        if torch.cuda.is_available():
            pos_weight = pos_weight.cuda()
        criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif opt.loss == 'asl':
        # Sử dụng Asymmetric Loss (ASL) - mặc định
        criterion_cls = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
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
        criterion_cls = FocalLoss(alpha=1, gamma=2)
    
    criterion_div = DistillKL(opt.kd_T)
    
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'itrd':
         opt.s_dim = feat_s[-1].shape[1]
         opt.t_dim = feat_t[-1].shape[1]
         opt.n_data = n_data
         criterion_kd = ITLoss(opt)
         module_list.append(criterion_kd.embed)
         trainable_list.append(criterion_kd.embed)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'ega':
        criterion_kd = EGA()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList([VIDLoss(s, t, t) for s, t in zip(s_n, t_n)])
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        pass
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'rrd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = RRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # Classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # Other knowledge distillation loss

    #=====================optimizer=====================
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(trainable_list.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(trainable_list.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(trainable_list.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # Append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    #=====================cuda=====================
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    #=====================eval teacher=====================
    teacher_auc, teacher_loss = validate(val_loader, model_t, criterion_cls, opt)
    print(f'Teacher Avg. AUC: {teacher_auc:.4f}')

    #=====================routine=====================
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> Training...")

        time1 = time.time()
        train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('Epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.add_scalar('train_loss', train_loss, epoch)

        val_auc, val_loss = validate(val_loader, model_s, criterion_cls, opt)

        # logger.add_scalar('val_auc', val_auc, epoch)
        # logger.add_scalar('val_loss', val_loss, epoch)

        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_auc': best_auc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('Saving the best model!')
            torch.save(state, save_file)
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

        # Regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'val_auc': val_auc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    print('==> Best student Avg. AUC:', best_auc)

    # Save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)
    # logger.close()


def init(model_s, model_t, init_modules, criterion, train_loader, opt):
    """ Initialization """
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.learning_rate
        
    optimizer = optim.SGD(init_modules.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
                    contrast_idx = contrast_idx.cuda()

            # ==============forward===============
            preact = (opt.distill == 'abound')
            feat_s, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplementedError('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # ===================print======================
        # logger.add_scalar('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()


def train(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """ One epoch distillation """
    # Set modules as train()
    for module in module_list:
        module.train()
        
    # Set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        # Kiểm tra xem model có phải là ViT không
        is_vit = 'vit' in opt.model_s.lower()
        
        if is_vit:
            feat_s, logit_s = model_s(input, is_feat=True)
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True)
        else:
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
        
        feat_t = [f.detach() for f in feat_t]

        # Classification (CE) + KL div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # Other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'itrd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'ega':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        elif opt.distill == 'rrd' and opt.memory_type == 'fifo':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd' and opt.memory_type == 'fifo':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'rrd' and opt.memory_type == 'momentum':
             f_s = feat_s[-1]
             f_t = feat_t[-1]
             loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'crd' and opt.memory_type == 'momentum':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================print======================
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    print(' * Loss {loss.avg:.4f}'.format(loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, opt):
    """ Validation """
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # Collect all outputs and targets for AUC calculation
    all_outputs = []
    all_targets = []

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            # Handle different data formats
            if len(data) == 2:
                input, target = data
            elif len(data) == 3:
                input, target, index = data
            elif len(data) == 4:
                input, target, index, contrast_idx = data
            else:
                raise ValueError(f"Unexpected data format with {len(data)} elements")

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Collect outputs and targets for AUC calculation
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())

            # Measure loss
            losses.update(loss.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses))

    # Calculate AUC
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    aucs, avg_auc = compute_auc_multilabel(all_outputs, all_targets)
    
    # Print AUC per class
    class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                   'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
                   'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    print('\nAUC per class:')
    for i, (class_name, auc) in enumerate(zip(class_names, aucs)):
        print(f'{class_name}: {auc:.4f}')
    
    print(f'\n * Avg. AUC: {avg_auc:.4f}')

    return avg_auc, losses.avg


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """ Learning rate schedule according to RotNet """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """ Sets the learning rate to the initial LR decayed by decay rate every steep step """
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """ Computes and stores the average and current value """
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


def accuracy_multilabel(output, target, topk=(1,)):
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
