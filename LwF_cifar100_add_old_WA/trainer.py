import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR

import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import pickle
from dataset import BatchData
from model import PreResNet
# from model1 import VGG16
from cifar import Cifar100
from exemplar import Exemplar
from copy import deepcopy
import torch.backends.cudnn as cudnn


class Trainer:
    def __init__(self, total_cls):
        cudnn.benchmark = True
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = Cifar100()
        self.model = PreResNet(32,total_cls).cuda()
        # self.model = VGG16(total_cls)
        # print(self.model)
        # self.model = nn.DataParallel(self.model, device_ids=[0])
        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)


    def test(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)

        previous_model = None

        dataset = self.dataset
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num : {inc_i}")
            train, test = dataset.getNextClasses(inc_i)
            print(len(train), len(test))
            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_ys.extend(train_y)

            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=(2e-4/(inc_i+1)))
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

            exemplar.update(total_cls//dataset.batch_num, (train_x, train_y))

            # self.seen_cls = 20 + inc_i *20
            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)

            test_acc = []
            ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(self.seen_cls-20, self.seen_cls, inc_i)
            if os.path.exists(ckp_name):
                self.model = torch.load(ckp_name)
            else:
                for epoch in range(epoches):
                    print("---"*50)
                    print("Epoch", epoch)
                    scheduler.step()
                    cur_lr = self.get_lr(optimizer)
                    print("Current Learning Rate : ", cur_lr)
                    self.model.train()
                    if inc_i > 0:
                        self.stage1_distill(train_data, criterion, optimizer)
                    else:
                        self.stage1(train_data, criterion, optimizer)
                    acc = self.test(test_data)
                    print(self.model.fc.weight)
                print(self.model.fc.weight)
                # 与BIC的不同在于，他发现了参数的意义，调参的过程，WA不用二次训练的
                if inc_i >0 :
                    # 保持所有的参数不变
                    self.model.eval()
                    # 旧类别的权重
                    weight_old = self.model.fc.weight.data[:self.seen_cls-20]
                    # 新类别的权重
                    weight_new = self.model.fc.weight.data[self.seen_cls-20:self.seen_cls]
                    # 计算二范数
                    Norm_weight_old = torch.sum(torch.norm(weight_old,dim=1)) / torch.norm(weight_old,dim=1).shape[0]
                    Norm_weight_new = torch.sum(torch.norm(weight_new,dim=1)) / torch.norm(weight_new,dim=1).shape[0]
                    lanbuda = Norm_weight_old/Norm_weight_new
                    # 计算好以后重新赋值
                    self.model.fc.weight.data[:self.seen_cls-20] = weight_old
                    self.model.fc.weight.data[self.seen_cls-20:self.seen_cls] = weight_new * (lanbuda)
                    self.model.cuda()
                    print(torch.norm(self.model.fc.weight.data,dim=1))

                torch.save(self.model, ckp_name)
            print(self.model.fc.weight)
            self.previous_model = deepcopy(self.model)
            acc = self.test(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)

    def get_one_hot(self, target,num_class):
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
        one_hot=one_hot.scatter_(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            loss = criterion(p[:,:self.seen_cls], label)

            # for abc in self.model.fc.parameters():
            #     abc.data.clamp_(min=0)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            for abc in self.model.fc.parameters():
                abc.data.clamp_(min=1)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            
            with torch.no_grad():
                pre_p = self.previous_model(image)
                
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            # loss_hard_target = nn.CrossEntropyLoss()(p[:,self.seen_cls-20:self.seen_cls], label-self.seen_cls+20)
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target +  loss_hard_target
            
            # for abc in self.model.fc.parameters():
            #     abc.data.clamp_(min=5)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            for abc in self.model.fc.parameters():
                abc.data.clamp_(min=1)

            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))



