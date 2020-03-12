import torch
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import random
from model import *
from attacks import Attacker, PGD_L2, DDN
import time
import torch.nn.functional as F

import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    help='model architecture')
parser.add_argument('--depth', default=110, type=int,
                    help='depth for resnet')
parser.add_argument('--save_path', type=str, default='./check_points')
parser.add_argument('--check_point', type=str, default='ckpt.t7')
parser.add_argument('--gauss_num', default=4, type=int,
                    help='Number of Gaussian samples per input')
parser.add_argument('--sigma', default=0.25, type=float,
                    metavar='W', help='initial sigma for each data')
parser.add_argument('--steps', default=10, type=int,
                    help='iteration steps for PGD attack')


def main():
    global args
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.save_path
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model = resnet110()

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # print("created model with configuration: %s", model_config)
    print("run arguments: %s", args)
    with open(model_path + '/log_{}.txt'.format(args.check_point), 'a') as f:
        f.writelines(str(args) + '\n')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("number of parameters: {}".format(num_parameters))
    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([transforms.ToTensor(),])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    if os.path.exists(os.path.join(model_path, args.check_point)):  # , 'Error: no results directory found!'
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(model_path, args.check_point))
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict(checkpoint['net'])

    for epsilon in np.arange(0.25, 2.0, 0.25):
        attacker = PGD_L2(steps=args.steps, device=device, max_norm=epsilon)
        end = time.time()

        correct = 0
        data_size = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = len(inputs)
            data_size += batch_size
            model.eval()
            # model.parameters().requires_grad_(False)
            inputs = inputs.repeat(1, args.gauss_num, 1, 1).reshape([batch_size * args.gauss_num] + list(inputs[0].shape))
            targets_tmp = targets.unsqueeze(1).repeat(1, args.gauss_num).reshape(-1, 1).squeeze()
            noise = torch.randn_like(inputs)

            with torch.enable_grad():
                inputs = attacker.attack(model, inputs, targets_tmp, noise=noise)

            noisy_inputs = inputs + noise

            outputs = model(noisy_inputs)

            soft_max = F.softmax(outputs, dim=1).view(batch_size, args.gauss_num, -1).mean(1)
            predict = soft_max.argmax(1)

            correct += predict.eq(targets).sum().item()

        acc = correct / data_size
        with open(model_path + '/log.txt', 'a') as f:
            f.writelines('The accuracy on adversarial sample for epsilon {} is {}'.format(epsilon, acc))
        print('elasped time is {}'.format(time.time() - end))
        print('The accuracy on adversarial sample for epsilon {} is {}'.format(epsilon, acc))


main()