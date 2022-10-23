import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from myNetwork import network
from iCIFAR100 import iCIFAR100
from utils import mean, std
from TinyImageNet import TinyImageNet


class ProtoRE:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.numssl = 4 if 'S' in args.ref else 1
        self.model = network(args.fg_nc * self.numssl, feature_extractor)
        self.radius = None
        self.prototype = None
        self.sup_prototype = None
        self.class_label = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.classes = []
        self.device = device
        self.old_model = None
        self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.24705882352941178),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean[args.data_name],
                                                                        std[args.data_name])])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean[args.data_name],
                                                                       std[args.data_name])])
        if args.data_name == 'cifar100':
            self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
            self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        else:
            self.train_dataset = TinyImageNet('./dataset', transform=self.train_transform)
            self.test_dataset = TinyImageNet('./dataset', test_transform=self.test_transform, train=False)
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass-self.task_size, self.numclass]
        self.classes.append(classes)
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(self.numclass * self.numssl)
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        print(len(self.train_dataset))
        for epoch in range(self.epochs):
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                if 'S' in self.args.ref:
                    # self-supervised learning based label augmentation
                    images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(self.numssl)], 1)
                    images = images.view(-1, 3, 32, 32)
                    target = torch.stack([target * self.numssl + k for k in range(self.numssl)], 1).view(-1)

                opt.zero_grad()
                loss = self._compute_loss(images, target, old_class)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
            scheduler.step()

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs[:, ::self.numssl]  # only compute predictions on original class nodes
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, old_class=0):
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output/self.args.temp, target)
        if self.old_model is None:
            return loss_cls
        else:
            if self.args.ref == 'L':
                cur_target = target - old_class
                loss_cls = nn.CrossEntropyLoss()(output[:, old_class:]/self.args.temp, cur_target)

                old_outputs = self.old_model(imgs)
                soft_target = F.softmax(old_outputs / 2.0, dim=1)
                logp = F.log_softmax(output[:, :old_class] / 2.0, dim=1)
                loss_kd = -torch.mean(torch.sum(soft_target * logp, dim=1))
            else:
                feature = self.model.feature(imgs)
                feature_old = self.old_model.feature(imgs)
                loss_kd = torch.dist(feature[-1], feature_old[-1], 2)

            proto_label = []
            proto_length = len(target) // self.numssl
            index = np.random.randint(0, old_class, proto_length)
            proto_label.extend(self.numssl * (np.array(self.class_label))[index])
            proto_label = torch.from_numpy(np.asarray(proto_label, dtype=np.int64)).to(self.device)

            loss_proto = 0
            if 'P' in self.args.ref:
                proto = []
                if 'A' in self.args.ref:
                    disturbance = np.random.normal(0, 1, [proto_length, 512]) * self.radius
                else:
                    disturbance = 0
                proto.extend([np.array(self.prototype)[index] + disturbance])
                proto = np.reshape(proto, (proto_length, -1))
                proto = torch.from_numpy(np.float32(np.asarray(proto))).float().to(self.device)

                # Randomly mix the mean prototypes of the same task.
                if 'M' in self.args.ref:
                    reminder_proto = []
                    reminder_label = []
                    for classes in self.classes[:-1]:
                        cur_index = np.array([i for i in range(proto_length)])
                        cur_index = cur_index[(index >= classes[0]) == (index < classes[1])]
                        if len(cur_index) % 2 == 1:
                            reminder_proto.append(proto[cur_index[0]].cpu().numpy())
                            reminder_label.append(proto_label[cur_index[0]].cpu().numpy())
                            cur_index = cur_index[1:]
                        if len(cur_index) > 1:
                            alpha = 0.5
                            # alpha = np.random.randint(0, 10) / 10      # random hybrid coefficient for ablation study
                            length = int(len(cur_index) // 2)
                            proto_aug_l = proto[cur_index][:length]
                            proto_aug_r = proto[cur_index][length:]
                            proto_label_l = proto_label[cur_index][:length]
                            proto_label_r = proto_label[cur_index][length:]
                            proto_aug_mix = alpha * proto_aug_l + (1 - alpha) * proto_aug_r
                            soft_feat_mix = self.model.fc(proto_aug_mix)
                            loss_proto += alpha * nn.CrossEntropyLoss()(soft_feat_mix / self.args.temp, proto_label_l) + \
                                          (1 - alpha) * nn.CrossEntropyLoss()(soft_feat_mix / self.args.temp,
                                                                              proto_label_r)
                    if len(reminder_proto) > 0:
                        reminder_proto_aug = np.array(reminder_proto, dtype=np.float32)
                        reminder_label = np.array(reminder_label, dtype=np.int64)
                        reminder_proto_aug = torch.tensor(reminder_proto_aug).to(self.device)
                        reminder_label = torch.tensor(reminder_label).to(self.device)
                        soft_feat_reminder = self.model.fc(reminder_proto_aug)
                        loss_proto += nn.CrossEntropyLoss()(soft_feat_reminder / self.args.temp, reminder_label)

                else:
                    soft_feat_aug = self.model.fc(proto)
                    loss_proto = nn.CrossEntropyLoss()(soft_feat_aug / self.args.temp, proto_label)

            loss_protoSup = 0
            if 'E' in self.args.ref:
                proto_sup = []
                proto_sup.extend(np.array(self.sup_prototype)[index][:, np.random.randint(0, self.args.comp_size), :])
                proto_sup = np.array(proto_sup, dtype=np.float32)
                proto_sup = np.reshape(proto_sup, (proto_length, -1))
                proto_sup = torch.from_numpy(np.asarray(proto_sup)).float().to(self.device)
                soft_feat_sup = self.model.fc(proto_sup)
                loss_protoSup = nn.CrossEntropyLoss()(soft_feat_sup / self.args.temp, proto_label)

            return loss_cls + self.args.kd_weight*loss_kd + \
                   self.args.proto_weight*loss_proto + self.args.protoSup_weight*loss_protoSup

    def afterTrain(self, current_task):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)

        if current_task == 0:
            filename = self.args.save_path + self.args.data_name + '/' + '%d_model' % self.args.fg_nc
            if 'S' in self.args.ref:
                filename = filename + '_S.pkl'
            else:
                filename = filename + '.pkl'
        else:
            filename = path + '%d_model.pkl' % self.numclass
        torch.save(self.model, filename)

        if 'P' in self.args.ref:
            self.protoSave(current_task)

        if 'E' in self.args.ref:
            self.protoSup(current_task)

        self.class_label = [i for i in range(self.numclass)]
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()
        self.numclass += self.task_size

    def protoSave(self, current_task):
        print('save mean prototype ...')
        features = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(self.train_loader):
                feature = self.model.feature(images.to(self.device))
                labels.extend(target.numpy())
                features.extend(feature[-1].cpu().numpy())

        labels_set = np.unique(labels)
        labels = np.array(labels)
        features = np.array(features)
        feature_dim = features.shape[1]
        prototype = []
        radius = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            print(f'radius: {self.radius}')
        else:
            self.prototype = np.concatenate((self.prototype, prototype), axis=0)

    def protoSup(self, current_task):
        print("save supplementary prototype ...")
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(self.train_loader):
                feature = self.model.feature(images.to(self.device))
                labels.extend(target.numpy())
                features.extend(feature[-1].cpu().numpy())
        labels = np.array(labels)
        labels_set = np.unique(labels)
        features = np.array(features)
        sup_prototype = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            with torch.no_grad():
                length = len(feature_classwise)
                correct = torch.tensor([0 for _ in range(length)])
                K = 50       # random disturbance factor
                gamma = 0.5  # the coefficient of disturbance
                np.random.seed(self.args.seed)
                for _ in range(K):
                    cur_feature = feature_classwise + gamma * np.random.normal(0, 1, 512)
                    cur_feature = torch.tensor(cur_feature, dtype=torch.float32).to(self.device)
                    outputs = self.model.fc(cur_feature)
                    predicts = torch.max(outputs, dim=1)[1]
                    targets = torch.tensor([item for _ in range(length)]).to(self.device)
                    correct[predicts.cpu() == targets.cpu()] += 1
                cur_prototype = []
                order = torch.sort(correct).indices
                for i in range(self.args.comp_size):
                    cur_prototype.append(feature_classwise[order[i]])
            sup_prototype.append(cur_prototype)

        if current_task == 0:
            self.sup_prototype = sup_prototype
        else:
            self.sup_prototype = np.concatenate((self.sup_prototype, sup_prototype), axis=0)



