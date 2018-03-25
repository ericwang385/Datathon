import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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
    cnn = CNN()
    cnn.cuda()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_function = nn.MSELoss()

    xs = np.load("xs.npy")
    ys = np.load("ys.npy")
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=42)
    X_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
    y_test = torch.from_numpy(np.array(y_test, dtype=np.float32))
    X_test = Variable(X_test).cuda()
    y_test = Variable(y_test).cuda()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(zip(X_train, y_train)):
            x = torch.from_numpy(np.array([x], dtype=np.float32))
            y = torch.from_numpy(np.array([y], dtype=np.float32))
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()

            output = cnn(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
            # if 1:
                test_output = cnn(X_test)
                loss_test = loss_function(test_output, y_test)
                print('Epoch:', epoch, '|Step:', step,
                      '|train loss:%.4f' % loss.data[0], '|test loss:%.4f' % loss_test.data[0])

        torch.save(cnn, "friends.tf")