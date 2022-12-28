import os
import argparse
from sklearn import manifold

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch

from utils import utils
from robustness import datasets, defaults, model_utils
from utils import fine_tunify, transfer_datasets

import numpy as np
from models import resnet

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def scatter(x, y):
    tx = scale_to_01_range(x[:, 0])
    ty = scale_to_01_range(x[:, 1])

    palette = np.array(sns.color_palette("hls",10)[::-1])
    f = plt.figure(figsize = (15, 12))
    ax = plt.subplot(aspect='equal')
    ax.scatter(tx, ty, c = palette[y])
    # ax.axis('off')
    # ax.axis('tight')

    plt.title("Visualization of ImageNet10class")
    # legend with color patches
    handles = []
    for i in range(10):
        patch = mpatches.Patch(color=palette[i], label=i)
        handles.append(patch)
        
    plt.legend(handles=handles, fontsize=10, loc=4)
    # plt.show()
    plt.savefig("resnet50_autoencoder_representation.jpg")

    # return f, ax

def computeTSNEProjectionOfLatentSpace(data_loader, model, args, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualization of Imagenet '
    
    feat_encodes = np.empty((2048, ), dtype='float')
    img_labels = np.empty((1, ), dtype='int')

    model.eval()
    for it, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        images = images.cuda()
        x, feat, recon_feat = model(images)
        feat = feat.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        feat_encodes = np.vstack((feat_encodes, feat))
        img_labels = np.hstack((img_labels, labels))
        
    feat_encodes = feat_encodes[1:]
    img_labels  = img_labels [1:]


    print(feat_encodes.shape)
    print(img_labels.shape)
    
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, perplexity=50, n_iter=1000, learning_rate=200,
                   n_iter_without_progress=10, random_state=0)
    X_tsne = tsne.fit_transform(feat_encodes)
    print(X_tsne[:, 1].shape)
    print(X_tsne[:, 0].shape)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        scatter(X_tsne, img_labels)
        # plt.show()
    else:
        return X_tsne

def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    args.batch_size = args.batch_size_per_gpu
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8)
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

    return ds, train_loader, validation_loader

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Visualize model's representation on ImageNet")

    ## dataset parameters
    parser.add_argument('--data', type = str, default = "data/ILSVRC2012_img_train_10cls_100shot_1k", help = '')
    parser.add_argument('--dataset', type = str, default = 'imagenet' ,help = '')
    parser.add_argument('--eval_only', action='store_true', help = '')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')

    ## model 
    parser.add_argument('--model_path', type = str, default = "teacher_weights/StandardSP-ImageNet1k-ResNet50.pth", help='')
    args = parser.parse_args()

    
    ds, train_loader, _ = get_dataset_and_loaders(args)
    model = resnet.resnet50_AE_1d(pretrained=True, initpath=args.model_path).cuda()

    computeTSNEProjectionOfLatentSpace(train_loader, model, args)