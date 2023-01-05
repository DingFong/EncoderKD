import argparse
import os
import cox.store

import numpy as np
import torch

from robustness import datasets, defaults, model_utils
import train_distill
from robustness.tools import helpers

from torch import nn
from torchvision import models
import torch

from utils import constants as cs
from utils import fine_tunify, transfer_datasets
from models import resnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
def main(args, store):
    '''Given arguments and a cox store, trains as a model. Check out the 
    argparse object in this file for argument options.
    '''
    import time

    starttime = time.asctime(time.localtime(time.time()))
    print("start: ", starttime)
    ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    student_model = resnet.resnet18_feat_pre_relu()

    update_params = freeze_model(student_model, freeze_level=args.freeze_level)
    print(f"Dataset: {args.dataset} | Model: {args.student_arch}")
    if args.teacher_path:

        teacher_model = resnet.resnet50_AE_1d(pretrained=True, initpath=args.teacher_path).cuda()
        train_distill.train_model(args, student_model, (train_loader, validation_loader), teacher_model=teacher_model, store=store,
                      checkpoint=None, update_params=update_params)


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size_per_gpu, workers=16)
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
            only_val=args.eval_only, batch_size=args.batch_size, workers=16)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, 8, args.subset)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = torch.tensor([0., 0., 0.])
            new_ds.std = torch.tensor([1., 1., 1.])
            ds = new_ds
    return ds, train_loader, validation_loader

def freeze_model(model, freeze_level):
    '''
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    '''
    # Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            # if not freeze and f'layer{freeze_level}' not in name:
            if not freeze and f'layer{freeze_level}' not in name and 'emb' not in name and 'l2norm' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Train student")

    parser.add_argument('--student_arch', type = str, default = "resnet18", help = '')
    parser.add_argument('--data', type = str, default = "data/imagenet", help = '')
    parser.add_argument('--dataset', type=str, default = 'imagenet', help='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--resume', action='store_true', help='')
    parser.add_argument('--pytorch-pretrained', action='store_true', help='')
    parser.add_argument('--out_dir', type=str, default="student_weights", help='')
    parser.add_argument('--exp_name', type=str, default="resnet18", help='weight path')
    parser.add_argument('--eval_only', action='store_true', help = '')
    parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
   

    # training
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument("--step-lr", default=30, type=int, help='')
    parser.add_argument("--step_lr_gamma", default=0.1, type=float, help='')
    parser.add_argument("--momentum", type = float, default = 0.9)
    parser.add_argument("--weight_decay", type = float, default = 5e-4)

    
    # amp
    parser.add_argument('--mixed_precision',action='store_true', help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")

    # --- optimizer: adabelief
    parser.add_argument('--optimizer_custom', type=str, default='sgd', help='optimizer') 
    ## adv
    parser.add_argument('--adv_train', action='store_true', help='')
    parser.add_argument('--adv_eval', action='store_true', help='')

    # add
    parser.add_argument('--weight', action='store_true',
                        help='Whether to put weights or not')
    parser.add_argument('--weight_path', type=str, default=None, help='weight path')
    parser.add_argument('--abandom_fc', action='store_true', default=False,
                        help='Whether abandom fc after leaned mask to ft')

    # scheduler
    parser.add_argument('--max_lr', type=float, default=1.0, help='1cycle.')
    parser.add_argument('--custom_lr_schedule', type=str, default=None, help='custom_lr_schedule.')  

    # KD
    parser.add_argument('--KD_T', type=float, default=4, help='Temperature in Knowledge Distillation.')
    parser.add_argument('--KD_T_mid', type=float, default=4, help='Temperature in Knowledge Distillation.')
    parser.add_argument('--cluster_KD_T', type=float, default=4, help='Temperature in Knowledge Distillation.')
    parser.add_argument('--KD_a', type=float, default=0.0, help='Balancing weight between losses in KD.')
    parser.add_argument('--loss_w', type=float, default=20, help='Balancing weight between losses in KD.')
    parser.add_argument('--KD_c', type=float, default=0.0, help='Balancing weight between losses in KD.')
    parser.add_argument('--KD_r', type=float, default=0.0, help='Balancing weight between losses in KD.')
    parser.add_argument('--KD_multi', type=float, default=0.0, help='Balancing weight between losses in KD.')
    parser.add_argument('--threshold', type=float, default=0.7, help='entropy threshold for teacher logit to select confident samples.')
    parser.add_argument('--teacher_path', type=str, default="teacher_weights/resnet_ae/checkpoint0075.pth", help='Teacher model path for Knowledge Distillation.')
    parser.add_argument('--cluster_KD_mode', type=str, default='euc', help='euc or cos distance.')
    parser.add_argument('--teacher_arch', type=str, default='resnet50', help='Teacher model arch for Knowledge Distillation.')
    parser.add_argument('--featureKD', action='store_true',help='Whether to use feature KD')
    parser.add_argument('--relationKD', action='store_true',help='Whether to use instance relation KD')
    parser.add_argument('--clusterKD', action='store_true',help='Whether to use cluster KD')
    parser.add_argument('--RKD', action='store_true',help='Whether to use RKD, cvpr19')
    parser.add_argument('--FSP', action='store_true',help='Whether to use FSP')
    parser.add_argument('--overhaul', action='store_true',help='Whether to use overhaul')
    parser.add_argument('--multi', action='store_true',help='Whether to use multi position FKD')

    #-- save ckpt at certain epochs
    parser.add_argument('--save_epoch_list', nargs='+', help='Set save ckpt epoch', required=False)
    parser.add_argument('--saveckp_freq', type = int, default = 5, help='')

    parser.add_argument('--dp_device_ids', type=int, nargs="+", default=None, help="")

    args = parser.parse_args()
    
    # Create store and log the args
    store = cox.store.Store(args.out_dir, args.exp_name)
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')
    main(args, store)