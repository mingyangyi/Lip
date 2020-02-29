from __future__ import print_function

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import time
import logging
import torchvision
import pickle
import torchvision.transforms as transforms
import numpy as np
import random
from utils import *
from macer import macer_train
from rs.certify import certify
from rs.test import test
from model import resnet110

import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    help='model architecture')
parser.add_argument('--depth', default=110, type=int,
                    help='depth for resnet')
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--data_dir', type=str, default='//gcr_share/data/imagenet/2012')
parser.add_argument('--task', default='train',
                    type=str, help='Task: train or test')
##########################################################################
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--epochs', default=450, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.1, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: None)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--resume', default='True', type=str, help='resume from checkpoint')
###############################################################################
parser.add_argument('--training_method', default='macer', type=str, metavar='training method',
                    help='The method of training')
parser.add_argument('--gauss_num', default=16, type=int,
                    help='Number of Gaussian samples per input')
parser.add_argument('--sigma', default=0.25, type=float,
                    metavar='W', help='initial sigma for each data')
parser.add_argument('--lam', default=12.0, type=float,
                    metavar='W', help='initial sigma for each data')
parser.add_argument('--gamma', default=8.0, type=float, help='Hinge factor')
parser.add_argument('--beta', default=16.0, type=float, help='Inverse temperature of softmax (also used in test)')


def main():
    global args
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.save = args.optimizer + '_' + args.model + '_' + args.dataset + '_' + args.training_method + '_' + \
                str(args.lr) + '_' + str(args.sigma) + '_' + str(args.lam) + '_' + str(args.gamma) + '_' + str(args.beta)
    save_path = os.path.join(args.save_path, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.info("creating model %s", args.model)
    model = resnet110()

    if device == 'cuda':
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # print("created model with configuration: %s", model_config)
    print("run arguments: %s", args)
    with open(save_path+'/log.txt', 'a') as f:
        f.writelines(str(args) + '\n')

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("number of parameters: {}".format(num_parameters))
    # Data
    print('==> Preparing data..')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5070588235294118, 0.48666666666666664, 0.4407843137254902),
            #                      (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5070588235294118, 0.48666666666666664, 0.4407843137254902),
            #                      (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)),
        ])

    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5),
            #                      (0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0:wq.5),
            #                      (0.5)),
        ])

    else:
        raise ValueError('No such dataset')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=args.lr_decay_ratio)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.epochs <= 200:
            scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=args.lr_decay_ratio)
        else:
            scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=args.lr_decay_ratio)


    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    elif args.dataset == 'imagenet':
        print('Loading data from zip file')
        train_dir = os.path.join(args.data_dir, 'train.zip')
        valid_dir = os.path.join(args.data_dir, 'validation.zip')
        print('Loading data into memory')

        trainset = InMemoryZipDataset(train_dir, transform_train, 32)
        testset = InMemoryZipDataset(valid_dir, transform_test, 32)

        print('Found {} in training data'.format(len(trainset)))
        print('Found {} in validation data'.format(len(validset)))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=4)

    else:
        raise ValueError('There is no such dataset')

    if args.resume == 'True':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if os.path.exists(save_path + '/ckpt.t7'):#, 'Error: no results directory found!'
            checkpoint = torch.load(save_path + '/ckpt.t7')
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            scheduler.step(start_epoch)

    num_classes = 10
    train_vector = []

    if args.task == 'train':
        for epoch in range(start_epoch, args.epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            print('create an optimizer with learning rate as:', lr)
            model.train()
            start_time = time.time()
            c_loss, r_loss, acc = macer_train(args.training_method, args.sigma, args.lam, args.gauss_num, args.beta,
                                              args.gamma, num_classes, model, trainloader,
                                              optimizer, device)

            print('Training time for each epoch is %g, optimizer is %s, model is %s' % (
                time.time() - start_time, args.optimizer, args.model + str(args.depth)))

            if args.epochs >= 200:
                if epoch % 50 == 0 and epoch >= 400:
                    # Certify test
                    print('===test(epoch={})==='.format(epoch))
                    t1 = time.time()
                    model.eval()

                    test(model, device, testloader, num_classes, mode='both', sigma=args.sigma,
                         beta=args.beta, file_path=(None if save_path is None else os.path.join(save_path,
                                                                                                'test_accuracy.txt')))

                    certify(model, device, testset, num_classes,
                            mode='hard', start_img=500, num_img=500, skip=1,
                            sigma=args.sigma, beta=args.beta,
                            matfile=(None if save_path is None else os.path.join(save_path,
                                                                                 'certify_radius{}.txt'.format(
                                                                                     epoch))))

                    t2 = time.time()
                    print('Elapsed time: {}'.format(t2 - t1))

            else:
                if epoch % 50 == 0 and epoch >= 100:
                    # Certify test
                    print('===test(epoch={})==='.format(epoch))
                    t1 = time.time()
                    model.eval()

                    test(model, device, testloader, num_classes, mode='both', sigma=args.sigma,
                         beta=args.beta, file_path=(None if save_path is None else os.path.join(save_path,
                                                                                                'test_accuracy.txt')))
                    certify(model, device, testset, num_classes,
                            mode='hard', start_img=500, num_img=500, skip=1,
                            sigma=args.sigma, beta=args.beta,
                            matfile=(None if save_path is None else os.path.join(save_path,
                                                                                 'certify_radius{}.txt'.format(
                                                                                     epoch))))

                    t2 = time.time()
                    print('Elapsed time: {}'.format(t2 - t1))

            print('\n Epoch: {0}\t'
                         'Cross Entropy Loss {c_loss:.4f} \t'
                         'Robust Loss {r_loss:.3f} \t'
                         'Total Loss {loss:.4f} \t'
                         'Accuracy {acc:.4f} \n'
                         .format(epoch + 1, c_loss=c_loss, r_loss=r_loss, loss=c_loss - r_loss,
                                 acc=acc))

            with open(save_path + '/log.txt', 'a') as f:
                f.write(str('\n Epoch: {0}\t'
                         'Cross Entropy Loss {c_loss:.4f} \t'
                         'Robust Loss {r_loss:.3f} \t'
                         'Accuracy {acc:.4f} \t'
                         'Total Loss {loss:.4f} \n'
                         .format(epoch + 1, c_loss=c_loss, r_loss=r_loss,
                                 acc=acc, loss=c_loss- r_loss)) + '\n')

            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                # 'trainset': trainset
            }

            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(state, save_path + '/ckpt.t7')
            if epoch % 10 == 0:
                torch.save(state, save_path + '/{}.t7'.format(epoch))
            if os.path.exists(save_path + '/train_vector'):
                with open(save_path + '/train_vector', 'rb') as fp:
                    train_vector = pickle.load(fp)
            train_vector.append([epoch, c_loss, r_loss, acc])
            with open(save_path + '/train_vector', 'wb') as fp:
                pickle.dump(train_vector, fp)

    else:

        test(model, device, testloader, num_classes, mode='both', sigma=args.sigma,
             beta=args.beta, file_path=(None if save_path is None else os.path.join(save_path,
                                                                                    'test_accuracy.txt')))

        certify(model, device, testset, num_classes,
                mode='hard', start_img=500, num_img=500, skip=1,
                sigma=args.sigma, beta=args.beta,
                matfile=(None if save_path is None else os.path.join(save_path,
                                                                     'certify_radius.txt')))


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(888)
    main()
