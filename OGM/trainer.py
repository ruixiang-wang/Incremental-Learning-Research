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
from model1 import seresnet34
from cifar import Cifar100
# from exemplar import Exemplar
from copy import deepcopy
import torch.backends.cudnn as cudnn

import scipy.spatial.distance as spd
import libmr

from sklearn.metrics import confusion_matrix
class Trainer:
    def __init__(self, total_cls):
        cudnn.benchmark = True
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = Cifar100()
        self.model1 = PreResNet(32,20).cuda()
        self.model = seresnet34().cuda()
        # print(self.model)
        # self.model = nn.DataParallel(self.model, device_ids=[0])
        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                # transforms.RandomCrop(64,padding=4),
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

    def test2(self, testdata2, pred_index,inc_i):
        print("test data number : ",len(testdata2))
        self.model.eval()
        self.linshi_model1.eval()
        self.linshi_model2.eval()
        count = 0
        correct = 0
        wrong = 0
        num_index = 0
        target_real = []
        target_pred = []
        a = 0
        b = 0
        crorrect1 = list(0. for i in range (20+inc_i*20))
        total = list(0. for i in range(20+inc_i*20))
        for i, (image, label) in enumerate(testdata2):
            image = image.cuda()
            label = label.view(-1).cuda()
            if label.cpu().numpy()<self.seen_cls-20:
                p = self.linshi_model1(image) # old 
                pred = p[:,:self.seen_cls].argmax(dim=-1)
                if pred == label:
                    a = a+1
            else:
                p = self.linshi_model2(image)
                pred = p[:,:self.seen_cls].argmax(dim=-1)
                if pred == label:
                    b = b+1
            if inc_i >= 0 :
                res = pred == label
                for label_index in range(len(label)):
                    label_single = label[label_index]
                    crorrect1[label_single] += res[label_index].item()
                    total[label_single] += 1
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
            if i==0:
                target_real = label.cpu().numpy()
                target_pred = pred.cpu().numpy()
            else:
                target_real = np.hstack((target_real,label.cpu().numpy()))
                target_pred = np.hstack((target_pred,pred.cpu().numpy()))
            num_index = num_index+1
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        print("a = ",a,"b = ",b)
        self.cm_plot(target_real,target_pred,inc_i)
        self.model.train()
        print("---------------------------------------------")
        return acc

    def cm_plot(self, original_label, predict_label,pic=None):
        cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
        plt.figure()
        plt.matshow(cm, cmap=plt.cm.jet)     # 画混淆矩阵，配色风格使用cm.Blues
        plt.colorbar()    # 颜色标签
        plt.ylabel('True class')  # 坐标轴标签
        plt.xlabel('Predicted class')  # 坐标轴标签
        # plt.title('confusion matrix')
        if pic is not None:
            plt.savefig(str(pic) + '.jpg')

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()

        previous_model = None
        dataset = self.dataset
        test_xs = []
        test_ys = []

        test_accs = []
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num : {inc_i}")
            train, test = dataset.getNextClasses(inc_i)
            print(len(train), len(test))
            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            train_data = DataLoader(BatchData(train_x, train_y, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            test_data2 = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=1, shuffle=False)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
            optimizer1 = optim.SGD(self.model1.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler1 = StepLR(optimizer1, step_size=70, gamma=0.1)

            self.seen_cls = 20 + inc_i *20
            print("seen cls number : ", self.seen_cls)

            test_acc = []
            ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(self.seen_cls-20, self.seen_cls, inc_i)
            ckp_name1 = './checkpoint1/{}_run_{}_iteration_{}_model1.pth'.format(self.seen_cls-20, self.seen_cls, inc_i)
            # train CL_model-model1
            if inc_i > 0:
                if os.path.exists(ckp_name1):
                    self.model1 = torch.load(ckp_name1)
                else:
                    for epoch in range(220):
                        print("---"*50)
                        print("epoch = ",epoch)
                        scheduler1.step()
                        self.model1.train()
                        self.stage_new(train_data, criterion, optimizer1)
                    torch.save(self.model1, ckp_name1)          
                # calculate openmax
                scores, labels = [], []
                self.model1.eval()
                with torch.no_grad():
                    for m, (image, label) in enumerate(test_data2):
                        image = image.cuda()
                        label = label.view(-1).cuda()
                        outputs,feature = self.model1(image)
                        scores.append(outputs)
                        labels.append(label)
                scores = torch.cat(scores,dim=0).cpu().numpy()
                labels = torch.cat(labels,dim=0).cpu().numpy()
                scores = np.array(scores)[:, np.newaxis, :]
                labels = np.array(labels)

                # Openmax
                _, mavs, dists = self.compute_av(train_data)
                categories = list(range(0, 20))
                weibull_tail = 20 #
                weibull_alpha = 3
                weibull_threshold = 0.9
                weibull_model = self.fit_weibull(mavs, dists, categories, weibull_tail, "euclidean")
                
                pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
                for score in scores:
                    so, ss = self.openmax(weibull_model, categories, score,
                                     0.5, weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
                    pred_softmax.append(np.argmax(ss))
                    pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= weibull_threshold else 20)
                    pred_openmax.append(np.argmax(so) if np.max(so) >= weibull_threshold else 20)
                # acc = self.test3(test_data2,accept_value)
                print("Evaluation...")
                labels1 = labels - self.seen_cls+20 
                labels2 = np.where(labels1>=0,labels1,20) #old categories:20 new categories:0-19
                labels3 = np.where(labels2>=20,labels2,0) #old categories:20 new categories:0

                pred_softmax_threshold1 = np.array(pred_softmax_threshold)
                pred_softmax_threshold2 = np.where(pred_softmax_threshold1>=20,pred_softmax_threshold1,0)

                pred_softmax_threshold2 = list(pred_softmax_threshold2)

                pred_index = []
                for k in range(len(pred_softmax_threshold2)):
                    if pred_softmax_threshold2[k]==20:
                        pred_index.append(k)

                print("len(pred_index):",len(pred_index))


            # train IL_model-model
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
                torch.save(self.model, ckp_name)
            self.previous_model = deepcopy(self.model)
            # change fc
            if inc_i > 0:
                self.linshi_model1 = deepcopy(self.model)  #to the old class
                self.linshi_model1.eval()
                weight_old = self.linshi_model1.linear.weight.data[:self.seen_cls-20]
                weight_new = self.linshi_model1.linear.weight.data[self.seen_cls-20:self.seen_cls]
                self.linshi_model1.linear.weight.data[:self.seen_cls-20] = weight_old * 10
                self.linshi_model1.linear.weight.data[self.seen_cls-20:self.seen_cls] = weight_new 
                self.linshi_model1.cuda()

                self.linshi_model2 = deepcopy(self.model)  #to the new class
                self.linshi_model2.eval()
                weight_old = self.linshi_model2.linear.weight.data[:self.seen_cls-20]
                weight_new = self.linshi_model2.linear.weight.data[self.seen_cls-20:self.seen_cls]
                self.linshi_model2.linear.weight.data[:self.seen_cls-20] = weight_old
                self.linshi_model2.linear.weight.data[self.seen_cls-20:self.seen_cls] = weight_new * 10
                self.linshi_model2.cuda()
            if inc_i == 0:
                acc = self.test(test_data)
            else:   
                acc = self.test2(test_data2,pred_index,inc_i)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage_new(self, train_data, criterion, optimizer1):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model1(image)[0]
            loss = criterion(p[:,:20], label-self.seen_cls+20)
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        # alpha = (self.seen_cls - 2)/ self.seen_cls
        # print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            # image = torch.from_numpy(np.concatenate((image, image, image), axis=-3))
            p = self.model(image)
            
            with torch.no_grad():
                pre_p = self.previous_model(image)
                
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,self.seen_cls-20:self.seen_cls], label-self.seen_cls+20)
            # loss_hard_target = nn.CrossEntropyLoss()(p[:,self.seen_cls-20:self.seen_cls], label)
            loss = loss_soft_target +  loss_hard_target
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))

    # openmax's function
    #############################################
    #########       Don't change      ###########
    #############################################

    def openmax(self,weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
        """Re-calibrate scores via OpenMax layer
        Output:
            openmax probability and softmax probability
        """
        nb_classes = len(categories)

        ranked_list = input_score.argsort().ravel()[::-1][:alpha]
        alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
        omega = np.zeros(nb_classes)
        omega[ranked_list] = alpha_weights

        scores, scores_u = [], []
        for channel, input_score_channel in enumerate(input_score):
            score_channel, score_channel_u = [], []
            for c, category_name in enumerate(categories):
                mav, dist, model = self.query_weibull(category_name, weibull_model, distance_type)
                channel_dist = self.calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
                wscore = model[channel].w_score(channel_dist)
                modified_score = input_score_channel[c] * (1 - wscore * omega[c])
                score_channel.append(modified_score)
                score_channel_u.append(input_score_channel[c] - modified_score)

            scores.append(score_channel)
            scores_u.append(score_channel_u)

        scores = np.asarray(scores)
        scores_u = np.asarray(scores_u)

        openmax_prob = np.array(self.compute_openmax_prob(scores, scores_u))
        softmax_prob = self.softmax(np.array(input_score.ravel()))
        return openmax_prob, softmax_prob

    def query_weibull(self, category_name, weibull_model, distance_type='eucos'):
        return [weibull_model[category_name]['mean_vec'],
                weibull_model[category_name]['distances_{}'.format(distance_type)],
                weibull_model[category_name]['weibull_model']]

    def compute_av(self, trainloader):
        scores = [[] for _ in range(20)]
        with torch.no_grad():
            for i, (image, label) in enumerate(trainloader):
                image, label = image.cuda(), label.cuda()
                p,feature = self.model1(image)
                for score, t in zip(p, label):
                    if torch.argmax(score) == t-self.seen_cls+20:
                        scores[t-self.seen_cls+20].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
        scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
        mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
        dists = [self.compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
        return scores, mavs, dists

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    

    def compute_openmax_prob(self, scores, scores_u):
        prob_scores, prob_unknowns = [], []
        for s, su in zip(scores, scores_u):
            channel_scores = np.exp(s)
            channel_unknown = np.exp(np.sum(su))

            total_denom = np.sum(channel_scores) + channel_unknown
            prob_scores.append(channel_scores / total_denom)
            prob_unknowns.append(channel_unknown / total_denom)

        # Take channel mean
        scores = np.mean(prob_scores, axis=0)
        unknowns = np.mean(prob_unknowns, axis=0)
        modified_scores = scores.tolist() + [unknowns]
        return modified_scores

    def compute_channel_distances(self, mavs, features, eu_weight=0.5):
        """
        Input:
            mavs (channel, C)
            features: (N, channel, C)
        Output:
            channel_distances: dict of distance distribution from MAV for each channel.
        """
        eucos_dists, eu_dists, cos_dists = [], [], []
        for channel, mcv in enumerate(mavs):  # Compute channel specific distances
            eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
            cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
            eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                                spd.cosine(mcv, feat[channel]) for feat in features])

        return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}
            
    def fit_weibull(self, means, dists, categories, tailsize=20, distance_type='eucos'):
        """
        Input:
            means (C, channel, C)
            dists (N_c, channel, C) * C
        Output:
            weibull_model : Perform EVT based analysis using tails of distances and save
                            weibull model parameters for re-adjusting softmax scores
        """
        weibull_model = {}
        for mean, dist, category_name in zip(means, dists, categories):
            weibull_model[category_name] = {}
            weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
            weibull_model[category_name]['mean_vec'] = mean
            weibull_model[category_name]['weibull_model'] = []
            for channel in range(mean.shape[0]):
                mr = libmr.MR()
                tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
                mr.fit_high(tailtofit, len(tailtofit))
                weibull_model[category_name]['weibull_model'].append(mr)

        return weibull_model

    def calc_distance(self,query_score, mcv, eu_weight, distance_type='eucos'):
        if distance_type == 'eucos':
            query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
                spd.cosine(mcv, query_score)
        elif distance_type == 'euclidean':
            query_distance = spd.euclidean(mcv, query_score)
        elif distance_type == 'cosine':
            query_distance = spd.cosine(mcv, query_score)
        else:
            print("distance type not known: enter either of eucos, euclidean or cosine")
        return query_distance
