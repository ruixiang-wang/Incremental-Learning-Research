import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import os
import sys
import numpy as np

from PRE import ProtoRE
from ResNet import resnet18_cbam
from iCIFAR100 import iCIFAR100
from utils import mean, std
from TinyImageNet import TinyImageNet


ref_cands = "LPASME"

parser = argparse.ArgumentParser(description='Prototype Representation Expansion for Incremental Learning')
parser.add_argument('--ref', default='L', type=str,
                    help='IL method. L: use LwF; P: use prototype; A: use protoAug; S: use SSL; '
                         'M: randomly mix mean prototype;'
                         'E: select supplement based anti-disturbance ability')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--numssl', default=1, type=int, help='weather use numssl')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--comp_size', default=50, type=int, help='the size of complete prototype')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--proto_weight', default=10.0, type=float, help='prototype loss weight')
parser.add_argument('--protoSup_weight', default=2.0, type=float, help='protoSup loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='training time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--seed', default=1993, type=int, help='the random seed of disturbance')
parser.add_argument('--shut_cls', default=20, type=int, help='when to shut down mix up')

args = parser.parse_args()


def main():
    if args.data_name == 'TinyImageNet':
        args.proto_weight = 5
        args.protoSup_weight = 1
    if 'S' in args.ref:
        args.numssl = 4
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '/' + str(args.fg_nc) + '_' + str(args.task_num) + 'x' + str(task_size) + '/' + f'ref={args.ref}'

    feature_extractor = resnet18_cbam()
    model = ProtoRE(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))

    print(args)
    for i in range(args.task_num+1):
        print(f"------> task: {i}")
        if i == 0:
            old_class = 0
            model_path = args.save_path + args.data_name + '/' + '%d_model' % args.fg_nc
            if 'S' in args.ref:
                model_path = model_path + '_S.pkl'
            else:
                model_path = model_path + '.pkl'
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
            if old_class - args.fg_nc == args.shut_cls and ('M' in args.ref or 'W' in args.ref) and 'A' in args.ref:
                print('shut down mix up!')
                args.ref = args.ref.replace('M', '')
            model_path = args.save_path + file_name + '/' + '%d_model.pkl' % (args.fg_nc + i * task_size)
        model.beforeTrain(i)
        if os.path.isfile(model_path):
            model.model = torch.load(model_path)
        else:
            model.train(old_class=old_class)
        model.afterTrain(i)


    ####### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean[args.data_name], std[args.data_name])])
    print("############# Test for each Task #############")
    if args.data_name == 'cifar100':
        test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    else:
        test_dataset = TinyImageNet('./dataset', test_transform=test_transform, train=False)
    acc_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        if current_task == 0:
            filename = args.save_path + args.data_name + '/' + '%d_model' % args.fg_nc
            if 'S' in args.ref:
                filename = filename + '_S.pkl'
            else:
                filename = filename + '.pkl'
        else:
            filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                outputs = outputs[:, ::args.numssl]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    print(acc_all)

    print("############# Test for up2now Task #############")
    if args.data_name == 'cifar100':
        test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    else:
        test_dataset = TinyImageNet('./dataset', test_transform=test_transform, train=False)
    stage_acc = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        if current_task == 0:
            filename = args.save_path + args.data_name + '/' + '%d_model' % args.fg_nc
            if 'S' in args.ref:
                filename = filename + '_S.pkl'
            else:
                filename = filename + '.pkl'
        else:
            filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size)
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            outputs = outputs[:, ::args.numssl]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print(accuracy)
        stage_acc.extend([accuracy])

    # compute Average ACC and Average FT
    print('\n')
    stage_acc = np.array(stage_acc)
    print(f'Total ACC: {stage_acc.tolist()}')
    print(f'Average ACC: {np.mean(stage_acc[1:])}')
    acc_all = np.array(acc_all)
    max_acc = np.max(acc_all[:-1], axis=0)[:-1]
    final_acc = acc_all[-1][:-1]
    print(f'Average FT: {np.mean(max_acc - final_acc)}')
    print('\n\n')


if __name__ == "__main__":
    main()
