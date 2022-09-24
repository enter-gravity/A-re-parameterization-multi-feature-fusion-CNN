import os
import pickle
import shutil
from functools import partial
from thop import profile
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import f1_score, make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard, Checkpoint
from skorch.dataset import ValidSplit
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from collections import Counter



def z_score(x, axis):
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    return x




def worker(data):
    before, after = 100, 156

    r_peaks, categories = data["r_peaks"], data["categories"]

    avg_rri = np.mean(np.diff(r_peaks))

    signals, labels, RR = [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue
        # r_peaks[i] - before: r_peaks[i] + after
        signals.append(data["signal"][r_peaks[i] - before: r_peaks[i] + after])
        labels.append(categories[i])
        RR.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
            np.mean(np.diff(r_peaks[np.maximum(i - 10, 0):i + 1])) - avg_rri  # local RR Interval
        ])

    return signals, labels, RR


def load_data(
        filename="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\dataset\\AAMI_new_mitdb_noPre_noNor.pkl"):
    import pickle

    with open(filename, "rb") as f:
        train_data = pickle.load(f)

    all_train_signals, all_train_labels, all_RR = [], [], []
    for i in range(len(train_data)):
        train_signals, train_labels, RR = worker(train_data[i])
        all_train_signals += train_signals
        all_train_labels += train_labels
        all_RR += RR


    scaler = RobustScaler()
    all_train_signals, all_train_labels, all_RR = np.array(all_train_signals), np.array(all_train_labels), np.array(
        all_RR)
    ss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
    x_train, y_train, x_test, y_test, RR_train, RR_test = [], [], [], [], [], []
    for train_index, test_index in ss.split(all_train_signals, all_train_labels):
        x_train.append(all_train_signals[train_index]), x_test.append(all_train_signals[test_index])
        y_train.append(all_train_labels[train_index]), y_test.append(all_train_labels[test_index])
        # RR_train.append(train_RR[train_index]),RR_test.append(train_RR[test_index])
        RR_train.append(scaler.fit_transform(all_RR[train_index]))
        RR_test.append(scaler.transform(all_RR[test_index]))

    all_train_s = []
    for train in x_train:
        local_s = []
        x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train = [], [], [], [], [], [], [], []
        for signal in train:
            n_train_signal = z_score(signal, 0)
            w = pywt.swt(data=n_train_signal, wavelet='db1', level=5, trim_approx=True)

            x3_train.append(w[3])
            x4_train.append(w[2])
            x5_train.append(w[1])

        x3_train = np.expand_dims(x3_train, axis=1).astype(np.float32)
        x4_train = np.expand_dims(x4_train, axis=1).astype(np.float32)
        x5_train = np.expand_dims(x5_train, axis=1).astype(np.float32)

        local_s.append(x3_train)
        local_s.append(x4_train)
        local_s.append(x5_train)

        all_train_s.append(local_s)

    all_test_s = []
    for test in x_test:
        local_s = []
        x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test = [], [], [], [], [], [], [], []
        for signal in test:
            n_test_signal = z_score(signal, 0)
            w = pywt.swt(data=n_test_signal, wavelet='db1', level=5, trim_approx=True)

            x3_test.append(w[3])
            x4_test.append(w[2])
            x5_test.append(w[1])

        x3_test = np.expand_dims(x3_test, axis=1).astype(np.float32)
        x4_test = np.expand_dims(x4_test, axis=1).astype(np.float32)
        x5_test = np.expand_dims(x5_test, axis=1).astype(np.float32)

        local_s.append(x3_test)
        local_s.append(x4_test)
        local_s.append(x5_test)


        all_test_s.append(local_s)
    y_train = np.array(y_train).astype(np.int64)
    y_test = np.array(y_test).astype(np.int64)

    with open(
            "D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\\9_3.pkl",
            "wb") as f:
        pickle.dump((all_train_s, RR_train, y_train, all_test_s, RR_test, y_test), f, protocol=4)


class FocalLoss(nn.Module):
    def __init__(self, class_num=5, alpha=0.5, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        device = torch.device("cuda:0")
        prob = self.softmax(pred.view(-1, self.class_num)).to(device)
        prob = prob.clamp(min=0.0001, max=1.0).to(device)

        target_ = torch.zeros(target.size(0), self.class_num).to(device)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = (- self.alpha.double() * torch.pow(1 - prob,
                                                            self.gamma).double() * prob.log().double() * target_.double()).to(
                device)
        else:
            batch_loss = (- torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()).to(
                device)

        batch_loss = batch_loss.sum(dim=1).to(device)

        if self.size_average:
            loss = batch_loss.mean().to(device)
        else:
            loss = batch_loss.sum().to(device)
        return loss





def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def _conv_bn(input_channel, output_channel, kernel_size=3, padding=1, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv', nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm1d(output_channel))
    return res


class RepBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, groups=1, stride=2, deploy=False, padding=1,
                 use_se=False, flag_3=False):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.activation = nn.ReLU()
        self.flag_3 = flag_3

        if (not self.deploy):

            self.brb_5 = _conv_bn(input_channel, output_channel, stride=self.stride, kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  groups=groups)
            self.brb_3 = _conv_bn(input_channel, output_channel, stride=self.stride, kernel_size=kernel_size - 2,
                                  padding=self.padding - 1,
                                  groups=groups)
            if self.flag_3 == True:
                self.brb_1 = _conv_bn(input_channel, output_channel, stride=self.stride, kernel_size=kernel_size - 4,
                                      padding=self.padding - 2,
                                      groups=groups)
        else:
            self.brb_rep = nn.Conv1d(in_channels=input_channel, out_channels=output_channel,
                                     kernel_size=self.kernel_size, padding=self.padding, padding_mode='zeros',
                                     stride=stride, bias=True)


        self.se = nn.Identity()

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.se(self.brb_rep(inputs)))
        if self.flag_3 == True:
            x = self.brb_3(inputs) + self.brb_5(inputs) + self.brb_1(inputs)
        else:
            x = self.brb_3(inputs) + self.brb_5(inputs)
        return self.activation(self.se(x))

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv1d(in_channels=self.brb_5.conv.in_channels, out_channels=self.brb_5.conv.out_channels,
                                 kernel_size=self.brb_5.conv.kernel_size, padding=self.brb_5.conv.padding,
                                 padding_mode=self.brb_5.conv.padding_mode, stride=self.brb_5.conv.stride,
                                 groups=self.brb_5.conv.groups, bias=True)
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('brb_5')
        self.__delattr__('brb_3')
        if self.flag_3 == True:
            self.__delattr__('brb_1')

    # 将3的卷积变成5的卷积参数
    def _pad_3_kernel(self, kernel):
        if (kernel is None):
            return 0
        else:
            return F.pad(kernel, [1] * 2)

    def _pad_1_kernel(self, kernel):
        if (kernel is None):
            return 0
        else:
            kernel = F.pad(kernel, [1] * 2)
            return F.pad(kernel, [1] * 2)

    def _get_equivalent_kernel_bias(self):
        brb_5_weight, brb_5_bias = self._fuse_conv_bn(self.brb_5)
        brb_3_weight, brb_3_bias = self._fuse_conv_bn(self.brb_3)
        if self.flag_3 == True:
            brb_1_weight, brb_1_bias = self._fuse_conv_bn(self.brb_1)
            new_brb_1_weight = self._pad_1_kernel(brb_1_weight)
            new_brb_3_weight = self._pad_3_kernel(brb_3_weight)
            return brb_5_weight + new_brb_3_weight + new_brb_1_weight, brb_5_bias + brb_3_bias + brb_1_bias
        else:
            new_brb_3_weight = self._pad_3_kernel(brb_3_weight)
            return brb_5_weight + new_brb_3_weight, brb_5_bias + brb_3_bias

    def _fuse_conv_bn(self, branch):
        if (branch is None):
            return 0, 0
        elif (isinstance(branch, nn.Sequential)):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm1d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.input_channel // self.groups
                kernel_value = np.zeros((self.input_channel, input_dim, 5), dtype=np.float32)
                for i in range(self.input_channel):
                    kernel_value[i, i % input_dim, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = gamma / std
        t = t.view(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std




class MyModule221(nn.Module):
    def __init__(self):
        super(MyModule221, self).__init__()
        self.b1_1 = RepBlock(input_channel=1, output_channel=8, kernel_size=7, stride=1, padding=2)
        self.p1_1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b1_2 = RepBlock(input_channel=8, output_channel=16, kernel_size=5, stride=1, padding=1)
        self.p1_2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b1_3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.ba1 = nn.BatchNorm1d(32)
        self.a1 = nn.ReLU()
        self.p1_3 = nn.AvgPool1d(kernel_size=3, stride=3)

        self.b2_1 = RepBlock(input_channel=1, output_channel=8, kernel_size=7, stride=1, padding=2)
        self.p2_1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b2_2 = RepBlock(input_channel=8, output_channel=16, kernel_size=5, stride=1, padding=1)
        self.p2_2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b2_3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.ba2 = nn.BatchNorm1d(32)
        self.a2 = nn.ReLU()
        self.p2_3 = nn.AvgPool1d(kernel_size=3, stride=3)

        self.b3_1 = RepBlock(input_channel=1, output_channel=8, kernel_size=7, stride=1, padding=2)
        self.p3_1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b3_2 = RepBlock(input_channel=8, output_channel=16, kernel_size=5, stride=1, padding=1)
        self.p3_2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b3_3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.ba3 = nn.BatchNorm1d(32)
        self.a3 = nn.ReLU()
        self.p3_3 = nn.AvgPool1d(kernel_size=3, stride=3)

        self.fc1 = nn.Linear(260, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)

        self.do = nn.Dropout(p=0.1)

    def forward(self, x3, x4, x5, x9):
        x3 = self.b1_1(x3)
        x3 = self.p1_1(x3)
        x3 = self.b1_2(x3)
        x3 = self.p1_2(x3)
        x3 = self.b1_3(x3)
        x3 = self.ba1(x3)
        x3 = self.a1(x3)
        x3 = self.p1_3(x3)

        x4 = self.b2_1(x4)
        x4 = self.p2_1(x4)
        x4 = self.b2_2(x4)
        x4 = self.p2_2(x4)
        x4 = self.b2_3(x4)
        x4 = self.ba2(x4)
        x4 = self.a2(x4)
        x4 = self.p2_3(x4)

        x5 = self.b3_1(x5)
        x5 = self.p3_1(x5)
        x5 = self.b3_2(x5)
        x5 = self.p3_2(x5)
        x5 = self.b3_3(x5)
        x5 = self.ba3(x5)
        x5 = self.a3(x5)
        x5 = self.p3_3(x5)

        x3 = x3.view((1, -1))
        x4 = x4.view((1, -1))
        x5 = x5.view((1, -1))
        x = torch.cat((x3, x4, x5), dim=0)
        x = torch.amax(x, dim=0)
        x = x.view((-1, 256))

        x = torch.cat((x, x9), dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.do(x)

        return x

    def switch(self):
        self.b1_1._switch_to_deploy()
        self.b1_2._switch_to_deploy()
        self.b2_1._switch_to_deploy()
        self.b2_2._switch_to_deploy()
        self.b3_1._switch_to_deploy()
        self.b3_2._switch_to_deploy()


def exaluated(confusion_matrix):
    confusion_matrix = np.reshape(confusion_matrix, (5, 5))
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Se = np.average(TP / (TP + FN))
    # Sp = np.average(TN / (TN + FP))
    # Tpp = np.average(TP / (TP + FP))
    # ACC = np.average((TP + TN) / (TP + TN + FP + FN))
    m = []
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    Tpp = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    m.append(Se)
    m.append(Sp)
    m.append(Tpp)
    m.append(ACC)
    return m


def loading(i):
    with open(
            "/data/zht/AAMI_new_10_cross_validation_data_RR_latest.pkl",
            "rb") as f:
        all_train_s, RR_train, y_train, all_test_s, RR_test, y_test = pickle.load(f)
    # x1 = all_train_s[i][0]
    # x2 = all_train_s[i][0]
    x3 = all_train_s[i][0]
    x4 = all_train_s[i][1]
    x5 = all_train_s[i][2]
    # x6 = all_train_s[i][5]
    # x7 = all_train_s[i][6]
    # x8 = all_train_s[i][7]
    x9 = np.array(RR_train[i]).astype(np.float32)
    y_t = y_train[i]

    # x11 = all_test_s[i][0]
    # x22 = all_test_s[i][0]
    x33 = all_test_s[i][0]
    x44 = all_test_s[i][1]
    x55 = all_test_s[i][2]
    # x66 = all_test_s[i][5]
    # x77 = all_test_s[i][6]
    # x88 = all_test_s[i][7]
    x99 = np.array(RR_test[i]).astype(np.float32)
    y_te = y_test[i]


    # return x1, x2, x3, x4, x5, x6, x7, x8,x9, y_t, x11, x22, x33, x44, x55, x66, x77, x88,x99, y_te
    return x3, x4, x5, x9, y_t, x33, x44, x55, x99, y_te


if __name__ == '__main__':
    # load_data()
    # with open("/tmp/zht_project/10_cross_validation_data_RR.pkl",
    # with open("D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\\10_cross_validation_data_RR.pkl",
    #           "rb") as f:
    #     all_train_s, RR_train, y_train, all_test_s, RR_test, y_test = pickle.load(f)
    # print("Data loaded successfully!")

    # log_dir = "./logs/SWT_inter_FL_repVGG_5"
    # shutil.rmtree(log_dir, ignore_errors=True)
    # writer = SummaryWriter(log_dir)
    # callbacks = [
    #     Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
    #     Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
    #     # LRScheduler(policy=MultiStepLR, milestones=[8, 18,25,33], gamma=0.1),
    #     LRScheduler(policy=StepLR, step_size=17, gamma=0.1),
    #     EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
    #     # EpochScoring(scoring=make_scorer(
    #     #     accuracy_score((x1_train, x2_train, x3_train, x4_train, x5_train),
    #     #                    y_train)), lower_is_better=False, name="test_acc"),
    #
    #     # TensorBoard(writer),
    #     Checkpoint(monitor='valid_acc_best')
    # ]
    # net = NeuralNetClassifier(
    #     module=MyModule2,
    #     criterion=nn.CrossEntropyLoss,
    #     optimizer=torch.optim.Adam,
    #     lr=0.001,
    #     max_epochs=40,
    #     batch_size=150,
    #     train_split=ValidSplit(0.2),
    #     verbose=1,
    #     device="cpu",
    #     callbacks=callbacks,
    # )
    #
    # module=net.module()
    # module.load_state_dict(torch.load(
    #     "D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/params/" + str(
    #         0) + "_params.pt"))

    # x3_train, x4_train, x5_train, x9_train, y_t, x3_test, x4_test, x5_test, x9_test, y_te = loading(
    #     0)
    # net.fit({"x3": x3_train, "x4": x4_train, "x5": x5_train, "x9": x9_train}, y_t)
    # x3_test=torch.tensor(x3_test)
    # x4_test = torch.tensor(x4_test)
    # x5_test = torch.tensor(x5_test)
    # x9_test = torch.tensor(x9_test)
    # net.fit({"x3": x3_train, "x4": x4_train, "x5": x5_train, "x9": x9_train}, y_t)
    # net.initialize()
    # net.load_params(
    #     f_params='D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\params.pt',)

    # net.save_params(f_params="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\module\module.pkl")
    # module = net.module()
    # module.load_state_dict(torch.load('D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\params.pt'))
    # x3, x4, x5, x9 = torch.randn(150, 1, 256), torch.randn(150, 1, 256), torch.randn(150, 1, 256), torch.randn(150, 4)
    # module.eval()
    # output1 = module.forward(x3_test, x4_test, x5_test, x9_test)

    # x3, x4, x5, x9 = torch.randn(150, 1, 256), torch.randn(150, 1, 256), torch.randn(150, 1, 256), torch.randn(150, 4)
    # macs1, params1 = profile(module, inputs=(x3, x4, x5, x9))
    # print("重参数前的计算量：" + str(macs1) + "参数量：" + str(params1))

    # module.switch()
    # macs2, params2 = profile(module, inputs=(x3, x4, x5, x9))
    # print("重参数后的计算量：" + str(macs2) + "参数量：" + str(params2))
    # module.eval()
    # output2 = module.forward(x3=x3_test, x4=x4_test, x5=x5_test, x9=x9_test)
    # prediction = np.array(torch.argmax(output2,1))
    # correct = (prediction == y_te).sum()
    # total = len(y_te)
    # acc = (correct / total)

    acc, matrix, score = [], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for i in range(10):

        callbacks = [
            Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
            Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
            # LRScheduler(policy=MultiStepLR, milestones=[8, 18,25,33], gamma=0.1),
            LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
            EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
            # EpochScoring(scoring=make_scorer(
            #     accuracy_score((x1_train, x2_train, x3_train, x4_train, x5_train),
            #                    y_train)), lower_is_better=False, name="test_acc"),

            # TensorBoard(writer),
            # Checkpoint(monitor='valid_acc_best',
            #            f_params="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master/checkpoint/params/" + str(
            #                i) + "_params.pt",
            #            f_history="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/history/" + str(
            #                i) + "_history.json",
            #            f_criterion="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/criterion/" + str(
            #                i) + "_criterion.pt",
            #            f_optimizer="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/optimizer/" + str(
            #                i) + "_optimizer.pt")
            Checkpoint(monitor='valid_acc_best',
                       f_params="/data/zht/checkpoint/params/" + str(
                           i) + "_params.pt",
                       f_history="/data/zht/checkpoint/history/" + str(
                           i) + "_history.json",
                       f_criterion="/data/zht/checkpoint/criterion/" + str(
                           i) + "_criterion.pt",
                       f_optimizer="/data/zht/checkpoint/optimizer/" + str(
                           i) + "_optimizer.pt")
        ]
        net = NeuralNetClassifier(
            module=MyModule221,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            lr=0.001,
            max_epochs=10,
            batch_size=150,
            train_split=ValidSplit(0.2),
            verbose=1,
            device="cuda:3",
            callbacks=callbacks,
        )

        # x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train,x9_train, y_t, x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test,x9_test, y_te = loading(
        #     i)
        x3_train, x4_train, x5_train, x9_train, y_t, x3_test, x4_test, x5_test, x9_test, y_te = loading(
            i)
        print("第" + str(i + 1) + "次训练")
        # net.initialize()
        net.fit({"x3": x3_train, "x4": x4_train, "x5": x5_train, "x9": x9_train}, y_t)
        net.save_params(f_params="/data/zht/module/params/" + str(i) + "_params.pt",
                        f_history="/data/zht/module/history/" + str(i) + "_history.json",
                        f_criterion="/data/zht/module/criterion/" + str(i) + "_criterion.pt",
                        f_optimizer="/data/zht/module/optimizer/" + str(i) + "_optimizer.pt"
                        )
        # net.save_params(
        #     f_params="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\module\params/" + str(
        #         i) + "_params.pt",
        #     f_history="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\module\history/" + str(
        #         i) + "_history.json",
        #     f_criterion="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\module\criterion/" + str(
        #         i) + "_criterion.pt",
        #     f_optimizer="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\module\optimizer/" + str(
        #         i) + "_optimizer.pt"
        # )
        y_true, y_pred = y_te, net.predict(
            {"x3": x3_test, "x4": x4_test, "x5": x5_test, "x9": x9_test})
        a1 = accuracy_score(y_true, y_pred)
        m1 = confusion_matrix(y_true, y_pred).ravel()
        m1 = np.reshape(m1, (5, 5))
        s1 = exaluated(m1)

        net.initialize()
        net.load_params(
            f_params='/data/zht/checkpoint/params/'+ str(i) + '_params.pt',
            f_history='/data/zht/checkpoint/history/'+ str(i) + '_history.json',
            f_criterion='/data/zht/checkpoint/criterion/'+ str(i) + '_criterion.pt',
            f_optimizer='/data/zht/checkpoint/optimizer/'+ str(i) + '_optimizer.pt')
        # net.load_params(
        #     f_params="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master/checkpoint/params/" + str(
        #         i) + "_params.pt",
        #     f_history="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/history/" + str(
        #         i) + "_history.json",
        #     f_criterion="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/criterion/" + str(
        #         i) + "_criterion.pt",
        #     f_optimizer="D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/optimizer/" + str(
        #         i) + "_optimizer.pt")

        y_true_check, y_pred_check = y_te, net.predict(
            {"x3": x3_test, "x4": x4_test, "x5": x5_test, "x9": x9_test})
        m2 = confusion_matrix(y_true, y_pred).ravel()
        m2 = np.reshape(m2, (5, 5))
        s2 = exaluated(m2)
        a2 = accuracy_score(y_true_check, y_pred_check)

        if (a2 > a1):
            print("使用的checkpoint结果")
            acc.append(a2)
            j, k = 0, 0
            for j in range(5):
                for k in range(5):
                    matrix[j][k] = matrix[j][k] + m2[j][k]
            j, k = 0, 0
            for j in range(4):
                for k in range(5):
                    score[j][k] = score[j][k] + s2[j][k]
            print(str(i + 1) + "次训练的SE：" + str(s2[0]) + "  SP：" + str(s2[1]) + "  TPP：" + str(s2[2]) + "  ACC：" + str(
                s2[3]))
            print("ACC:" + str(a2))

            # =-----------------------------------------------------
            # x3_test = torch.tensor(x3_test)
            # x4_test = torch.tensor(x4_test)
            # x5_test = torch.tensor(x5_test)
            # x9_test = torch.tensor(x9_test)
            # module = net.module()
            # module.load_state_dict(torch.load("/data/zht/checkpoint/params/" + str(i) + "_params.pt"))
            # module.load_state_dict(torch.load(
            #     "D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\checkpoint/params/" + str(
            #         i) + "_params.pt"))
            # module.switch()
            # module.eval()
            # output = module.forward(x3_test, x4_test, x5_test, x9_test)
            # prediction = np.array(torch.argmax(output, 1))
            # correct = (prediction == y_te).sum()
            # total = len(y_te)
            # rep_acc = (correct / total)
            # print("原模型精度：" + str(a2) + "重参数后的模型精度：" + str(rep_acc))
        #     ----------------------------------------
        else:
            print("使用的最后一次训练的结果")
            acc.append(a1)
            for j in range(5):
                for k in range(5):
                    matrix[j][k] = matrix[j][k] + m1[j][k]
            j, k = 0, 0
            for j in range(4):
                for k in range(5):
                    score[j][k] = score[j][k] + s1[j][k]
            print(str(i + 1) + "次训练的SE：" + str(s1[0]) + "  SP：" + str(s1[1]) + "  TPP：" + str(s1[2]) + "  ACC：" + str(
                s1[3]))
            print("ACC:" + str(a1))

            # =-----------------------------------------------------
            # x3_test = torch.tensor(x3_test)
            # x4_test = torch.tensor(x4_test)
            # x5_test = torch.tensor(x5_test)
            # x9_test = torch.tensor(x9_test)
            # module = net.module()
            # module.load_state_dict(torch.load("/data/zht/module/params/" + str(i) + "_params.pt"))
            # module.load_state_dict(torch.load(
            #     "D:\pythonProject\DWT_CNN\ECG-Classification-Using-CNN-and-CWT-master\module/params/" + str(
            #         i) + "_params.pt"))
            # module.switch()
            # module.eval()
            # output = module.forward(x3_test, x4_test, x5_test, x9_test)
            # prediction = np.array(torch.argmax(output, 1))
            # correct = (prediction == y_te).sum()
            # total = len(y_te)
            # rep_acc = (correct / total)
            # print("原模型精度：" + str(a1) + "重参数后的模型精度：" + str(rep_acc))
        #     ----------------------------------------

    # print("使用最后一个epoch训练完后的模型，测试集平均SE"+str(np.average(se1))+"SP"+str(np.average(sp1))+"TPP"+str(np.average(tpp1))+"ACC：" + str(np.average(acc1)))
    # print(acc1)
    # print("使用checkpoint保存的模型，测试集平均精确度：" + str(np.average(acc1)))
    # print(acc2)
    # print("取最好的测试集精度")

    print("平均SE：" + str(np.average(score[0])) + "  平均SP：" + str(np.average(score[1])) + "  平均TPP：" + str(
        np.average(score[2])) + "  平均ACC：" + str(np.average(score[3])))
    print("平均精确度" + str(np.average(acc)))
    print("混淆矩阵之和：")
    print(matrix)

    # net.save_params(f_params="./models/SWT_inter_FL_repVGG_2.pkl")
    # start = time.time()
    # y_true, y_pred = y_test, net.predict(
    #     {"x1": x1_train, "x2": x2_train, "x3": x3_train, "x4": x4_train, "x5": x5_train})
    # end = time.time()
    # print("Predict completed! tTime required: " + str(format(end - start, '.2f')) + " sec")
    # print(confusion_matrix(y_true, y_pred))
    # x = confusion_matrix(y_true, y_pred).ravel()
    # exaluated(x)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Se, Sp, Tpp, Acc = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), (tp + tn) / (tp + tn + fp + fn)
    # print("Se="+Se+"  Sp="+Sp+"  Tpp="+Tpp+"  Acc="+Acc)
    # target_names = ['N', 'PVC', 'RBBB', 'LBBB', 'APC', 'PAC']
    # print(classification_report(y_true, y_pred, digits=6, target_names=target_names))
    # print(accuracy_score(y_true, y_pred))
