import argparse
import os
import sys
import time
import datetime
import json
from pathlib import Path
import math
import numpy
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from robustness import datasets, defaults, model_utils
from utils import fine_tunify, transfer_datasets
from utils import utils

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from models import resnet



def train_one_epoch(model, criterion, train_loader, optimizer, 
                    lr_schedule, wd_schedule, epoch, fp16_scaler, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{args.epochs}]'


    for it, (images, _) in enumerate(metric_logger.log_every(train_loader, 10, header)):
    # for it, (images, _) in enumerate(tqdm(train_loader)):

        it = len(train_loader)* epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if  i == 0:
                param_group["lr"] = wd_schedule[it]

        # =========== Move data to GPU ===============
        # print('put img to gpu')
        # print(images)
        images =  torch.stack([im.cuda(non_blocking=True) for im in images])
        

        # =========== Training =================
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # print('fp16')
            x, feat, recon_feat, latent_code = model(images)
            # print(feat.shape)
            # print(feat)
            # print(recon_feat.shape)
            # print(recon_feat)
            # print(latent_code.shape)
            loss = criterion(feat, recon_feat)
            # print(loss)
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        
        # ============= Update ===============
        # print('backward')
        
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        # torch.cuda.synchronize()
    #     metric_logger.update(loss=loss.item())
    #     metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    #     metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    ## ============initial setting================
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    torch.backends.cudnn.benchmark = True


    ## =============== data preparation ============
    
    ds, train_loader = get_dataset_and_loaders(args)
    print(f"Data loaded: there are {len(ds)} images.")
    
    ## ====================== Model ============b=================

    model = resnet.resnet50_AE_1d(pretrained=True, initpath=args.model_path).cuda()
    if utils.has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=args.dp_device_ids, find_unused_parameters=True)
    # print(f"Model are built: they are both {args..model_path} network.")
    # time.sleep(20)

    ## =============== Loss ============================
    criterion = nn.MSELoss()

    ## =============== Optimizer =======================
    # optimizer = optim.SGD(model.parameters(), lr = args.lr)
    # ResNet part are freezed
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # ============== mixed precision training =============================
    fp16_scaler = None
    if args.use_fp16:
        print("use Fp16")
        fp16_scaler = torch.cuda.amp.GradScaler()
    

    ## ============== init schedulers =========================
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, 
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, 
        len(train_loader),
    )

    print(f"Loss, optimizer and schedulers ready.")

    ## ============ resume ====================================
    print(f"Checkpoint path : {args.out_dir}")
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.out_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model = model,
        optimizer=optimizer,
        criterion=criterion,
    )
    start_epoch = to_restore["epoch"]


    ## ============ Train =================================
    start_time = time.time()
    print('Start Training AutoEncoder')
    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # ============== training one epoch ================
        train_stats = train_one_epoch(model, 
                                    criterion, 
                                    train_loader, 
                                    optimizer, 
                                    lr_schedule, 
                                    wd_schedule,
                                    epoch,
                                    fp16_scaler,
                                    args
                                    )
                                    
        ## ============ write log =====================
        save_dict = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch+1,
            'args':args,
            'criterion':criterion.state_dict()
        }


        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.out_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.out_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    '''Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    '''
    print('[Resuming finetuning from a checkpoint...]')
    if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
        model, _ = model_utils.make_and_restore_model(
            arch=pytorch_models[args.arch](
                args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
            dataset=datasets.ImageNet(''), add_custom_forward=args.arch in pytorch_models.keys())
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(
            args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                               add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
    else:
        model, checkpoint = model_utils.make_and_restore_model(
            arch=args.arch, dataset=ds, resume_path=finetuned_model_path)
    return model, checkpoint

def get_model(args, ds, fc_classes=None):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to 
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')

    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    
    elif args.weight and os.path.isfile(args.weight_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, args.weight_path)
        checkpoint = None

    else:
        if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10 \
                and not args.resume_from_fc_of_target_data and not args.resume_from_diff_shape_fc:
            # import ipdb
            # ipdb.set_trace(context=20)
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            checkpoint = None

        elif args.resume_from_fc_of_target_data:
            model, _ = resume_finetuning_from_checkpoint(args,ds,args.model_path)
            checkpoint = None

        elif args.resume_from_diff_shape_fc:
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, fc_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=args.model_path,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
            return model, None

        else:
            model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds,
                                                          resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            # print(f'[Replacing the last layer with {args.additional_hidden} '
            #       f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            if fc_classes is None:
                print(f'[Replacing the last layer with {args.additional_hidden} '
                      f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
                model = fine_tunify.ft(
                    args.arch, model, ds.num_classes, args.additional_hidden)
            else:
                print(f'[Replacing the last layer with {args.additional_hidden} '
                      f'hidden layers and 1 classification layer that fc has dimension of {fc_classes}.]')
                model = fine_tunify.ft(
                    args.arch, model, fc_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')

    return model, checkpoint

def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    args.batch_size = args.batch_size_per_gpu
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_set, test_set = ds.get_dataset(only_val=False)
        # train_set = torch.utils.data.Subset(train_set, numpy.random.choice(len(train_set), 1000, replace=False)) # for test
        
        
        sampler = torch.utils.data.DistributedSampler(train_set, shuffle = True)
        data_loader = torch.utils.data.DataLoader(
            train_set,
            sampler = sampler,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = True,
        )

        # train_loader, validation_loader = ds.make_loaders(
        #     only_val=args.eval_only, batch_size=args.batch_size, workers=8)

    elif args.dataset == 'mix4':
        ds, train_loader, validation_loader = transfer_datasets.make_loaders_mix4(args.batch_size, 8)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = torch.tensor([0., 0., 0.])
            new_ds.std = torch.tensor([1., 1., 1.])
            ds = new_ds

    elif args.cifar10_cifar10:
        ds = datasets.CIFAR('/tmp')
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, 8, args.subset)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = torch.tensor([0., 0., 0.])
            new_ds.std = torch.tensor([1., 1., 1.])
            ds = new_ds

    return train_set, data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train teacher's autoencoder")
    parser.add_argument('--arch', type = str, default = "resnet50_AE_1d", help = '')
    parser.add_argument('--data', type = str, default = "data/imagenet", help = '')
    parser.add_argument('--model_path', type = str, default = "teacher_weights/StandardSP-ImageNet1k-ResNet50.pth", help='')
    parser.add_argument('--dataset', type = str, default = 'imagenet' ,help = '')
    parser.add_argument('--eval_only', action='store_true', help = '')
    parser.add_argument('--pytorch_pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
    parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
    parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
    
    parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
    parser.add_argument('--weight', action='store_true',
                        help='Whether to put weights or not')
    parser.add_argument('--weight_path', type=str, default=None, help='weight path')
    parser.add_argument('--fc_classes', type=int, default=1000, help='')
    parser.add_argument('--out_dir', type=str, default="teacher_weights/resnet_ae", help='')
    parser.add_argument('--exp_name', type=str, default="resnet50_AE_1d", help='weight path')
    
    
     # Training/Optimization parameters
    parser.add_argument('--use_fp16',action='store_false', help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    # parser.add_argument("--step-lr", default=30, type=int, help='')
    
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # S5.2
    parser.add_argument('--resume_from_fc_of_target_data', action='store_true', help='S5.2 resume from ckpt with fc of D2')
    parser.add_argument('--resume_from_diff_shape_fc', action='store_true', help='S5.2 resume from ckpt with fc of different shape with 1000 nor target data')
    parser.add_argument('--saveckp_freq', type = int, default = 5, help='')
    
    # Multiple GPU
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--dp_device_ids', type=int, nargs="+", default=None, help="")

    pytorch_models = {
        'resnet50_AE_1d': resnet.resnet50_AE_1d
    }
    
  

    args = parser.parse_args()
    main(args)
