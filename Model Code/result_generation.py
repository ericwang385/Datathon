import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import copy
from hoursblock import HourBlock
from pickle import dump

torch.manual_seed(1)

EPOCH = 10
LR = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=72 * 3 + 2, out_channels=512, kernel_size=7,
                      stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 128, 5, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, 5, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, 7),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(8, 2, 9),
            nn.Conv2d(2, 2, 1) # rectification layer
        )

    def forward(self, x):
        layers = [self.conv1, self.conv2, self.conv3,
                  self.conv4, self.conv5]
        for layer in layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    cnn = torch.load("model/f_9.tf")
    cnn.cuda()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_function = nn.MSELoss()

    xs = np.load("dataset_month1/xs.npy")
    ys = np.load("dataset_month1/ys.npy")

    start_date_time = datetime.datetime(2013, 1, 1, 0)
    current_date_time = datetime.datetime(2013, 1, 1, 0)
    delta = datetime.timedelta(hours=1)

    ground_truth = []
    predicted = []

    for step, (x, y) in enumerate(zip(xs, ys)):
        print(step)
        if step >= 7 * 24:
            break

        x = torch.from_numpy(np.array([x], dtype=np.float32))
        y = torch.from_numpy(np.array([y], dtype=np.float32))
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()

        output = cnn(b_x)

        ground_truth_i = HourBlock(current_date_time)
        ground_truth_matrix = b_y.cpu().data.numpy()[0]
        ground_truth_i.pick_up = ground_truth_matrix[0]
        ground_truth_i.drop_off = ground_truth_matrix[1]
        ground_truth.append(ground_truth_i)

        predicted_i = HourBlock(current_date_time)
        predicted_matrix = output.cpu().data.numpy()[0]
        predicted_i.pick_up = predicted_matrix[0]
        predicted_i.drop_off = predicted_matrix[1]
        predicted.append(predicted_i)

        current_date_time += delta

    dump(ground_truth, open("ground_truth.pkl", "wb+"))
    dump(predicted, open("predicted.pkl", "wb+"))