from __future__ import division, print_function

import sys
import time

import torch
# from .distiller_zoo import DistillKL2
import torch.nn as nn
import torch.nn.functional as F

from .util import AverageMeter, accuracy, reduce_tensor


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) 

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        
        input, target = batch_data
        
        data_time.update(time.time() - end)
        
        # input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        # output = model(input, is_feat=True)

        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    if opt.distill in ['scakd']:
        criterion_kd1 = criterion_list[3]
        
    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    # for idx, data in enumerate(train_loader):
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
            input_aug1 = input
            input_aug2 = input
        else:
            inputs, target = data
            input, input_aug1, input_aug2= inputs

            if opt.distill in ['scakd'] and input.shape[0] < opt.batch_size:
                continue

        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            input_aug1 = input_aug1.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            input_aug2 = input_aug2.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s= model_s(input, is_feat=True)
        aug_feat_s1, aug_logit_s1= model_s(input_aug1, is_feat=True)
        aug_feat_s2, aug_logit_s2= model_s(input_aug2, is_feat=True)


        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True)
            aug_feat_t1, aug_logit_t1 = model_t(input_aug1, is_feat=True)
            aug_feat_t2, aug_logit_t2 = model_t(input_aug2, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            aug_feat_t1 = [f.detach() for f in aug_feat_t1]
            aug_feat_t2 = [f.detach() for f in aug_feat_t2]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        # loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_div = criterion_div(logit_s, logit_t)
            loss_kd = 0
        elif opt.distill == 'gkd':
            loss_div = 0
            loss_div = criterion_kd(logit_s, logit_t)
            loss_kd = 0
        elif opt.distill in ['scakd']:
            loss_kd4 = criterion_kd(aug_logit_t1, aug_logit_s1, epoch) + criterion_kd(aug_logit_t2, aug_logit_s2, epoch) * opt.weight_SSL          # 最后一层
            logit_student = torch.cat((logit_s,aug_logit_s1,aug_logit_s2),dim=0)                                                      # 增强数据KD
            logit_teacher = torch.cat((logit_t, aug_logit_t1, aug_logit_t2), dim=0)
            loss_div = criterion_div(logit_student, logit_teacher) * 3.0
            feat_s_sum = []                                                                                                           # 自注意力
            feat_t_sum = []
            for i in range(len(feat_s) - 2):
               s = torch.cat((feat_s[i + 1], aug_feat_s1[i + 1], aug_feat_s2[i + 1]), dim=0)
               t = torch.cat((feat_t[i + 1], aug_feat_t1[i + 1], aug_feat_t2[i + 1]), dim=0)
               feat_s_sum.append(s)
               feat_t_sum.append(t)
            s_value, f_target, weight = module_list[1](feat_s_sum, feat_t_sum)
            loss_kd5 = criterion_kd1(s_value, f_target, weight) * opt.weight_e
            loss_kd = loss_kd4 + loss_kd5
        elif opt.distill == 'crd':
            loss_div = criterion_div(logit_s, logit_t)
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'similarity':
            loss_div = criterion_div(logit_s, logit_t)
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'pkt':
            loss_div = criterion_div(logit_s, logit_t)
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            loss_div = criterion_div(logit_s, logit_t)
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'srrl':
            loss_div = criterion_div(logit_s, logit_t)
            cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'dkd':
            loss_div = criterion_div(logit_s, logit_t)
            loss_kd = criterion_kd(logit_s, logit_t, target, opt.dkd_alpha, opt.dkd_beta, 4.0)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        loss_kl.update(loss_div.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        metrics = accuracy(logit_s, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses,
                top1=top1, top5=top5
                ))
            sys.stdout.flush()
            # print(temp)

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, opt):
    """validation"""
    
    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader)

    with torch.no_grad():
        # end = time.time()
        for idx, batch_data in enumerate(val_loader):

            input, target = batch_data


            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                        'GPU: {2}\t'
                        'Loss {loss.avg:.4f}\t'
                        'Acc@1 {top1.avg:.3f}\t'
                        'Acc@5 {top5.avg:.3f}'.format(
                        idx, n_batch, opt.gpu, loss=losses,
                        top1=top1, top5=top5))
    
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return [top1.avg, top5.avg, losses.avg]
