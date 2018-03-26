import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from cnn import CNN
from math import inf

torch.manual_seed(1)

EPOCH = 10
LR = 0.001

if __name__ == "__main__":
    cnn = CNN()
    cnn.cuda()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_function = nn.MSELoss()

    for epoch in range(EPOCH):
        best_test_loss = inf
        total_step = 0

        for month in range(1, 7):
            print("Epoch: %d, Month: %d" % (epoch, month))
            xs = np.load("dataset_month%d/xs.npy" % month)
            ys = np.load("dataset_month%d/ys.npy" % month)
            X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)

            for step, (x, y) in enumerate(zip(X_train, y_train)):
                total_step += 1
                x = torch.from_numpy(np.array([x], dtype=np.float32))
                y = torch.from_numpy(np.array([y], dtype=np.float32))
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()

                output = cnn(b_x)
                loss = loss_function(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mini_batch_loss = loss.data[0]

                if total_step % 1000 == 0:
                    test_loss = 0
                    for _step, (_x, _y) in enumerate(zip(X_test, y_test)):
                        _x = torch.from_numpy(np.array([_x], dtype=np.float32))
                        _y = torch.from_numpy(np.array([_y], dtype=np.float32))
                        b_x = Variable(_x).cuda()
                        b_y = Variable(_y).cuda()

                        output = cnn(b_x)
                        loss = loss_function(output, b_y)
                        test_loss += loss.data[0]

                    test_loss = test_loss / len(X_test)

                    print('Epoch:', epoch, '|Step:', step,
                          '|test loss:%.4f' % test_loss)

                    if test_loss < best_test_loss:
                        print("Saving!")
                        torch.save(cnn, "model/f_%d.tf" % (epoch))
                        best_test_loss = test_loss