import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from cnn import CNN

torch.manual_seed(1)

if __name__ == "__main__":
    # cnn = CNN()
    cnn = torch.load("friends.tf")
    cnn.cuda()
    print(cnn)
    xs = np.load("dataset_month1/xs.npy")
    ys = np.load("dataset_month1/ys.npy")
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=42)
    X_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
    y_test = torch.from_numpy(np.array(y_test, dtype=np.float32))
    X_test = Variable(X_test).cuda()
    y_test = Variable(y_test).cuda()

    for step, (x, y) in enumerate(zip(X_train, y_train)):
        x = torch.from_numpy(np.array([x], dtype=np.float32))
        y = torch.from_numpy(np.array([y], dtype=np.float32))
        # x = torch.FloatTensor(np.array([x]).tolist())
        # y = torch.FloatTensor(np.array([x]).tolist())
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()

        output = cnn(b_x)
        result = output.cpu().data.numpy()

        # np.set_printoptions(threshold=np.nan)
        result = np.array(result, dtype=np.int32)
        print(result)
        print(np.max(result), np.mean(result))

        b_y = b_y.cpu().data.numpy()
        print(b_y)
        print(np.max(b_y), np.mean(b_y))

        diff = b_y - result
        loss = np.sum((b_y - result) ** 2)
        print(loss)

        break