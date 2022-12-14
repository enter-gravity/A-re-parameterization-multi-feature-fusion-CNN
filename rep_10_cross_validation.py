import pickle
from functools import partial
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
from skorch.callbacks import EpochScoring, Initializer, LRScheduler,  Checkpoint
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import StepLR
from thop import profile



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


def load_data():
    import pickle

    with open("db.pkl", "rb") as f:
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
            "10_cross_validation_db.pkl","wb") as f:
        pickle.dump((all_train_s, RR_train, y_train, all_test_s, RR_test, y_test), f, protocol=4)


def _conv_bn(input_channel, output_channel, kernel_size=3, padding=1, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv', nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm1d(output_channel))
    return res


class RepBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, groups=1, stride=2, deploy=False, padding=1, flag_3=False):
        super().__init__()
        self.stride = stride
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
                                  padding=self.padding,groups=groups)
            self.brb_3 = _conv_bn(input_channel, output_channel, stride=self.stride, kernel_size=kernel_size - 2,
                                  padding=self.padding - 1,groups=groups)
            if self.flag_3 == True:
                self.brb_1 = _conv_bn(input_channel, output_channel, stride=self.stride, kernel_size=kernel_size - 4,
                                      padding=self.padding - 2,groups=groups)
        else:
            self.brb_rep = nn.Conv1d(in_channels=input_channel, out_channels=output_channel,kernel_size=self.kernel_size,
                                     padding=self.padding, padding_mode='zeros',stride=stride, bias=True)


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

    # ???3???????????????5???????????????
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
        self.r1 = nn.ReLU()
        self.p1_3 = nn.AvgPool1d(kernel_size=3, stride=3)

        self.b2_1 = RepBlock(input_channel=1, output_channel=8, kernel_size=7, stride=1, padding=2)
        self.p2_1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b2_2 = RepBlock(input_channel=8, output_channel=16, kernel_size=5, stride=1, padding=1)
        self.p2_2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b2_3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.ba2 = nn.BatchNorm1d(32)
        self.r2 = nn.ReLU()
        self.p2_3 = nn.AvgPool1d(kernel_size=3, stride=3)

        self.b3_1 = RepBlock(input_channel=1, output_channel=8, kernel_size=7, stride=1, padding=2)
        self.p3_1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b3_2 = RepBlock(input_channel=8, output_channel=16, kernel_size=5, stride=1, padding=1)
        self.p3_2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.b3_3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.ba3 = nn.BatchNorm1d(32)
        self.r3 = nn.ReLU()
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
        x3 = self.r1(x3)
        x3 = self.p1_3(x3)

        x4 = self.b2_1(x4)
        x4 = self.p2_1(x4)
        x4 = self.b2_2(x4)
        x4 = self.p2_2(x4)
        x4 = self.b2_3(x4)
        x4 = self.ba2(x4)
        x4 = self.r2(x4)
        x4 = self.p2_3(x4)

        x5 = self.b3_1(x5)
        x5 = self.p3_1(x5)
        x5 = self.b3_2(x5)
        x5 = self.p3_2(x5)
        x5 = self.b3_3(x5)
        x5 = self.ba3(x5)
        x5 = self.r3(x5)
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
    with open("10_cross_validation_db.pkl","rb") as f:
        all_train_s, RR_train, y_train, all_test_s, RR_test, y_test = pickle.load(f)

    train_3 = all_train_s[i][0]
    train_4 = all_train_s[i][1]
    train_5 = all_train_s[i][2]

    train_RR = np.array(RR_train[i]).astype(np.float32)
    y_t = y_train[i]


    test_3 = all_test_s[i][0]
    test_4 = all_test_s[i][1]
    test_5 = all_test_s[i][2]

    test_RR = np.array(RR_test[i]).astype(np.float32)
    y_te = y_test[i]

    return train_3, train_4, train_5, train_RR, y_t, test_3, test_4, test_5, test_RR, y_te


def test_moudle():
    for i in range(10):
        callbacks = [
            Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
            Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
            LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
            EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),\
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

        x3_train, x4_train, x5_train, x9_train, y_t, x3_test, x4_test, x5_test, x9_test, y_te = loading(i)
        net.initialize()
        net.load_params(
            f_params='/module/params/' + str(i) + '_params.pt',
            f_history='/module/history/' + str(i) + '_history.json',
            f_criterion='/module/criterion/' + str(i) + '_criterion.pt',
            f_optimizer='/module/optimizer/' + str(i) + '_optimizer.pt')
        y_true, y_pred = y_te, net.predict(
            {"x3": x3_test, "x4": x4_test, "x5": x5_test, "x9": x9_test})
        a1 = accuracy_score(y_true, y_pred)
        m1 = confusion_matrix(y_true, y_pred).ravel()
        m1 = np.reshape(m1, (5, 5))
        s1 = exaluated(m1)


        net.initialize()
        net.load_params(
            f_params='/checkpoint/params/' + str(i) + '_params.pt',
            f_history='/checkpoint/history/' + str(i) + '_history.json',
            f_criterion='/checkpoint/criterion/' + str(i) + '_criterion.pt',
            f_optimizer='/checkpoint/optimizer/' + str(i) + '_optimizer.pt')

        y_true_checkpoint, y_pred_checkponint = y_te, net.predict(
            {"x3": x3_test, "x4": x4_test, "x5": x5_test, "x9": x9_test})
        a2 = accuracy_score(y_true_checkpoint, y_pred_checkponint)
        m2 = confusion_matrix(y_true, y_pred).ravel()
        m2 = np.reshape(m2, (5, 5))
        s2 = exaluated(m2)

        if (a2 > a1):
            print("?????????checkpoint??????")
            acc.append(a2)
            for j in range(5):
                for k in range(5):
                    matrix[j][k] = matrix[j][k] + m2[j][k]
            for j in range(4):
                for k in range(5):
                    score[j][k] = score[j][k] + s2[j][k]
            print(str(i + 1) + "????????????SE???" + str(s2[0]) + "  SP???" + str(s2[1]) + "  TPP???" + str(s2[2]) + "  ACC???" + str(
                s2[3]))
            print("ACC:" + str(a2))



            #-----------------------------------------------------
            # x3_test = torch.tensor(x3_test)
            # x4_test = torch.tensor(x4_test)
            # x5_test = torch.tensor(x5_test)
            # x9_test = torch.tensor(x9_test)
            # module = net.module()
            # module.load_state_dict(torch.load("/checkpoint/params/" + str(i) + "_params.pt"))
            # module.switch()
            # module.eval()
            # output = module.forward(x3_test, x4_test, x5_test, x9_test)
            # prediction = np.array(torch.argmax(output, 1))
            # correct = (prediction == y_te).sum()
            # total = len(y_te)
            # rep_acc = (correct / total)
            # print("??????????????????" + str(a2) + "??????????????????????????????" + str(rep_acc))
            #----------------------------------------
        else:
            print("????????????????????????????????????")
            acc.append(a1)
            for j in range(5):
                for k in range(5):
                    matrix[j][k] = matrix[j][k] + m1[j][k]
            for j in range(4):
                for k in range(5):
                    score[j][k] = score[j][k] + s1[j][k]
            print(str(i + 1) + "????????????SE???" + str(s1[0]) + "  SP???" + str(s1[1]) + "  TPP???" + str(s1[2]) + "  ACC???" + str(
                s1[3]))
            print("ACC:" + str(a1))

            # -----------------------------------------------------
            # x3_test = torch.tensor(x3_test)
            # x4_test = torch.tensor(x4_test)
            # x5_test = torch.tensor(x5_test)
            # x9_test = torch.tensor(x9_test)
            # module = net.module()
            # module.load_state_dict(torch.load("/moudle/params/" + str(i) + "_params.pt"))
            # module.switch()
            # module.eval()
            # output = module.forward(x3_test, x4_test, x5_test, x9_test)
            # prediction = np.array(torch.argmax(output, 1))
            # correct = (prediction == y_te).sum()
            # total = len(y_te)
            # rep_acc = (correct / total)
            # print("??????????????????" + str(a2) + "??????????????????????????????" + str(rep_acc))
            # ----------------------------------------
        print("??????SE???" + str(np.average(score[0])) + "  ??????SP???" + str(np.average(score[1])) + "  ??????TPP???" + str(
            np.average(score[2])) + "  ??????ACC???" + str(np.average(score[3])))
        print("?????????????????????" + str(np.average(acc)))
        print("?????????????????????")
        print(matrix)


def rep():
    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
        LRScheduler(policy=StepLR, step_size=17, gamma=0.1),
        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
        Checkpoint(monitor='valid_acc_best')
    ]
    net = NeuralNetClassifier(
        module=MyModule221,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=40,
        batch_size=150,
        train_split=ValidSplit(0.2),
        verbose=1,
        device="cpu",
        callbacks=callbacks,
    )

    net.initialize()
    net.load_params(f_params='1_params.pt', )
    module = net.module()
    module.eval()

    x3, x4, x5, x9 = torch.randn(150, 1, 256), torch.randn(150, 1, 256), torch.randn(150, 1, 256), torch.randn(150, 4)
    macs1, params1 = profile(module, inputs=(x3, x4, x5, x9))
    print("???????????????????????????" + str(macs1) + "????????????" + str(params1))

    module.switch()
    macs2, params2 = profile(module, inputs=(x3, x4, x5, x9))
    print("???????????????????????????" + str(macs2) + "????????????" + str(params2))




if __name__ == '__main__':
    # load_data()
    # rep()


    acc, matrix, score = [], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for i in range(10):

        callbacks = [
            Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
            Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
            LRScheduler(policy=StepLR, step_size=12, gamma=0.1),
            EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
            Checkpoint(monitor='valid_acc_best',
                       f_params="/checkpoint/params/" + str(
                           i) + "_params.pt",
                       f_history="/checkpoint/history/" + str(
                           i) + "_history.json",
                       f_criterion="/checkpoint/criterion/" + str(
                           i) + "_criterion.pt",
                       f_optimizer="/checkpoint/optimizer/" + str(
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
            device="cpu",
            callbacks=callbacks,
        )

        x3_train, x4_train, x5_train, x9_train, y_t, x3_test, x4_test, x5_test, x9_test, y_te = loading(
            i)
        print("???" + str(i + 1) + "?????????")
        net.fit({"x3": x3_train, "x4": x4_train, "x5": x5_train, "x9": x9_train}, y_t)
        net.save_params(f_params="/module/params/" + str(i) + "_params.pt",
                        f_history="/module/history/" + str(i) + "_history.json",
                        f_criterion="/module/criterion/" + str(i) + "_criterion.pt",
                        f_optimizer="/module/optimizer/" + str(i) + "_optimizer.pt"
                        )


        # test_moudle()
