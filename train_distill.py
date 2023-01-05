import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid

from robustness.tools import helpers
from robustness.tools.helpers import AverageMeter, ckpt_at_epoch, has_attr
from robustness.tools import constants as consts

import os
import time
import warnings
from tqdm import tqdm

from KD import DistillKL, AngleLoss, DistillKL_mask
import torch.nn.functional as F

from utils.balance_dataparallel import BalancedDataParallel

try:
    from apex import amp
except Exception as e:
    warnings.warn('Could not import amp.')

def L2_loss_FKD(x, y, norm=False, exp_mode='exp', T=16, align=False):
    '''
    Compute L2 between two tensors
    '''
    # x: N x D
    # y: N x D

    if norm:
        x_n, y_n = x.norm(p=2, dim=1, keepdim=True), y.norm(p=2, dim=1, keepdim=True)
        x = x / (x_n.expand_as(x))
        y = y / (y_n.expand_as(y))

    if align:
        # import ipdb
        # ipdb.set_trace(context=20)
        cos_sim = F.cosine_similarity(x,y) # B
        x=x.permute(1,0)
        x = torch.sign(cos_sim) * x
        x = x.permute(1, 0)

    if exp_mode == 'exp':
        x = torch.exp(x / T)
        y = torch.exp(y / T)
    elif exp_mode == 'softmax':
        x = torch.nn.functional.softmax(x / T, dim=1)
        y = torch.nn.functional.softmax(y / T, dim=1)
        return torch.nn.functional.mse_loss(x, y, reduction='sum')

    return torch.nn.functional.mse_loss(x, y, reduction='mean')


def train_model(args, student_model, loaders, store, checkpoint = None, update_params = None, teacher_model=None, disable_no_grad = False, all_feat = False):

    ## Multiple GPU Training
    student_model = BalancedDataParallel(16, student_model.cuda(), device_ids=[0,1,2,3])
    
    
    # Logging setup
    loop_choice = _model_loop
    writer = store.tensorboard if store else None
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"


    ## initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, student_model, checkpoint, update_params)

    best_prec1, start_epoch = (0, 0)
    best_feat_relation_dis = 1000000
    best_logit_relation_dis = 1000000
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[prec1_key] if prec1_key in checkpoint \
            else loop_choice(args, 'val', val_loader, student_model, None, start_epoch - 1, args.adv_train, writer=None)[0]

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # if args.KD_r>0 and epoch == 30:
        #     args.KD_r = 100
        # train for one epoch
        train_prec1, train_loss = loop_choice(args, 'train', train_loader,
                                              student_model, opt, epoch, args.adv_train, writer, teacher_model=teacher_model, all_feat=all_feat)
        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model': student_model.state_dict(),
            'optimizer': opt.state_dict(),
            'schedule': (schedule and schedule.state_dict()),
            'epoch': epoch + 1,
            'amp': amp.state_dict() if args.mixed_precision else None,
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                              store.path, filename)
            torch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            ctx = torch.enable_grad() if disable_no_grad else torch.no_grad()
            with ctx:
                # prec1, nat_loss = loop_choice(args, 'val', val_loader, model,
                #                               None, epoch, False, writer, all_feat=all_feat)

                if args.epochs <200:
                    prec1, nat_loss = loop_choice(args, 'val', val_loader, student_model,
                                                                 None, epoch, False, writer,
                                                                 teacher_model=teacher_model, all_feat=all_feat)
                else:
                    prec1, nat_loss = 0, 0

            # loader, student_model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and loop_choice(args, 'val', val_loader,
                                                      student_model, None, epoch, True, writer, all_feat=all_feat)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            # log every checkpoint
            log_info = {
                'epoch': epoch + 1,
                'nat_prec1': prec1,
                'adv_prec1': adv_prec1,
                'nat_loss': nat_loss,
                'adv_loss': adv_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        # import ipdb
        # ipdb.set_trace(context=20)
        if args.save_epoch_list and str(epoch+1) in args.save_epoch_list: save_checkpoint(str(epoch+1)+'.ckpt')

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(student_model, log_info)

    return student_model



def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer, teacher_model=None, all_feat=False):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).
    Runs a single epoch of either training or evaluating.
    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        teacher_model: knowledge distillation teacher
    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = AverageMeter()
    loss_KD_feat_meter = AverageMeter()
    loss_KD_feat_mid_meter = AverageMeter()
    loss_KD_feat_rela_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    grad_norm_meter = AverageMeter()


    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()
    model = model.cuda()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
            if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
        else torch.nn.CrossEntropyLoss()

    if teacher_model:
        teacher_model.eval()
        KD_criterion = DistillKL(args.KD_T)
        KD_criterion_mid = DistillKL(args.KD_T_mid)
        # KD_criterion = DistillKL_mask(args.KD_T, args.threshold, args.fc_classes)

    angle_loss_criterion = AngleLoss()

    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = {
            'constraint': args.constraint,
            'eps': eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
            'random_start': args.random_start,
            'custom_loss': adv_criterion,
            'random_restarts': random_restarts,
            'use_best': bool(args.use_best)
        }

    iterator = tqdm(enumerate(loader), total=len(loader))
    current_lr = opt.state_dict()['param_groups'][0]['lr'] if opt is not None else 0

    for i, (inp, target) in iterator:
        # measure data loading time
        # import ipdb
        # ipdb.set_trace(context=20)
        target = target.cuda(non_blocking=True)
        inp = inp.cuda(non_blocking=True)

        output = model(inp)

        loss_KD_feat_mid = 0 if args.multi else torch.zeros(1).cuda()
        
        if teacher_model:
            
            # teacher_model = teacher_model.cuda(non_blocking=True)
            output_teacher = teacher_model(inp)


            # ====================== logit KD =======================================
            student_logits = output[0] if (type(output) is tuple) else output
            teacher_logits = output_teacher[0] if (type(output_teacher) is tuple) else output_teacher
            if args.KD_a >0:
                # import ipdb
                # ipdb.set_trace(context=20)
                loss_KD_logits = KD_criterion(student_logits, teacher_logits)
            else:
                loss_KD_logits = torch.zeros(1).cuda()
            # loss = (1-args.KD_a) * loss_cls + args.KD_a * loss_KD
            # ====================== logit KD =======================================

            # ====================== feature KD =======================================
            if args.loss_w >0:
                feat_s = output[1]
                feat_t = output_teacher[3]
                loss_KD_feat = L2_loss_FKD(feat_s, feat_t, norm=False, exp_mode='none', T=args.KD_T) # L2 loss for feat KD
            else:
                loss_KD_feat = torch.zeros(1).cuda()

            loss_KD_feat_meter.update(args.loss_w * loss_KD_feat.item())
            loss_KD_feat_mid_meter.update(args.KD_multi * args.loss_w * loss_KD_feat_mid.item())

            # ====================== feature KD =======================================



            # ====================== Student class loss =======================================
            if args.KD_c >0:
                loss_cls = train_criterion(student_logits, target)
                
            else:
                loss_cls = torch.zeros(1).cuda()
             # ====================== Student class loss =======================================

            loss = args.KD_a * loss_KD_logits + args.loss_w * loss_KD_feat + args.KD_c * loss_cls


        else:
            output = output[0] if (type(output) is tuple) else output
            loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
            else:
                prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
                # import ipdb
                # ipdb.set_trace(context=20)
                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except:
            warnings.warn('Failed to calculate the accuracy.')

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term = args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()

            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR

        # if has_attr(model.module.model.model, "feat_scale"):
        #     feat_scale = model.module.model.model.feat_scale
        # else:
        feat_scale = 1

        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Loss_feat: {feat:.3f} |'
                'Loss_feat_mid: {feat_mid:.3f} |'
                'Loss_feat_rela: {loss_feat_rela:.3f} |'
                # 'grad norm avg: {grad_norm:.3f}|'
                # 'grad norm val: {grad_norm_val:.3f}|'
                # 'KD_r: {KD_r}'
                'feat_scale: {fs:.3f}'
                'lr: {lr}||'.format(epoch, prec, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, feat=loss_KD_feat_meter.val, feat_mid=loss_KD_feat_mid_meter.val, loss_feat_rela=loss_KD_feat_rela_meter.val,
                                    grad_norm=grad_norm_meter.avg,grad_norm_val=grad_norm_meter.val,KD_r=args.KD_r, fs=feat_scale, lr=current_lr))

        if i == len(loader)-1:
            print('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Loss_feat: {feat:.3f} |'
                'Loss_feat_mid: {feat_mid:.3f} |'
                'Loss_feat_rela: {loss_feat_rela:.3f} |'
                # 'grad norm avg: {grad_norm:.3f}|'
                # 'grad norm val: {grad_norm_val:.3f}|'
                # 'KD_r: {KD_r}'
                'feat_scale: {fs:.3f}'
                'lr: {lr}||'.format(epoch, prec, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, feat=loss_KD_feat_meter.val, feat_mid=loss_KD_feat_mid_meter.val, loss_feat_rela=loss_KD_feat_rela_meter.val,
                                    grad_norm=grad_norm_meter.avg,grad_norm_val=grad_norm_meter.val,KD_r=args.KD_r, fs=feat_scale, lr=current_lr))


        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    # return top1.avg, losses.avg, feat_relation_dis_meter.avg, logit_relation_dis_meter.avg
    return top1.avg, losses.avg

def make_optimizer_and_schedule(args, model, checkpoint, params):
    """
    *Internal Function* (called directly from train_model)
    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.
    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params
    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    param_list = model.parameters() if params is None else params
    if args.optimizer_custom == 'sgd':
        optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_custom == 'adabelief':
        # pip install adabelief-pytorch==0.2.0
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(param_list, lr=args.lr, eps=1e-8, betas=(0.9, 0.999), weight_decouple=True, weight_decay=args.weight_decay,rectify=False)
    else:
        optimizer = None

    if args.mixed_precision:
        model.to('cuda')
        model, optimizer = amp.initialize(model, optimizer, 'O1')

    # Make schedule
    schedule = None
    if args.custom_lr_schedule == 'cyclic':
        eps = args.epochs
        lr_func = lambda t: np.interp([t], [0, eps * 4 // 15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_schedule == '1cycle':
        eps = args.epochs
        max_lr = args.max_lr
        # lr_func = lambda t: np.interp([t], [0, eps // 2, eps], [1, max_lr // args.lr, 1])[0]
        lr_func = lambda t: np.interp([t], [0, eps // 5 * 2, eps // 5 * 4, eps], [1, max_lr / args.lr, 1, 5e-5/args.lr])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_schedule == '3cycle':
        eps = args.epochs
        max_lr = args.max_lr
        # lr_func = lambda t: np.interp([t], [0, eps // 2, eps], [1, max_lr // args.lr, 1])[0]  # args.lr=0.03 or 0.05
        lr_func = lambda t: np.interp([t], [0, eps // 6 * 1, eps // 6 * 2, eps // 6 * 3, eps // 6 * 4, eps // 6 * 5, eps],
                                      [1, max_lr / args.lr, 1/10, 10, 1/100, 1, 5e-5/args.lr])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_schedule == 'multisteplr':
        args.milestones = [int(i) for i in args.milestones]
        schedule = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    elif args.custom_lr_schedule:
        cs = args.custom_lr_schedule
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == 'linear':
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()

        if 'amp' in checkpoint and checkpoint['amp'] not in [None, 'N/A']:
            amp.load_state_dict(checkpoint['amp'])

        # TODO: see if there's a smarter way to do this
        # TODO: see what's up with loading fp32 weights and then MP training
        if args.mixed_precision:
            model.load_state_dict(checkpoint['model'])

    return optimizer, schedule
