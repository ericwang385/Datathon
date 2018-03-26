import pickle
import numpy as np
from hoursblock import SPLITTING_COUNT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("month", type=int, help="Data set month")
parser.add_argument("--total_fraction", default=15, type=int, help="Total fraction, default 15")
args = parser.parse_args()

file_name = "dataset_month%d/%s.pkl" % (args.month, args.month)
data_list = pickle.load(open(file_name, "rb"))

iter_list = [iter(data_list) for _ in range(73)]
for i in range(len(iter_list)):
    for _ in range(len(iter_list) - i - 1):
        next(iter_list[i])
iter_list.reverse()

n = SPLITTING_COUNT

xs = []
ys = []

for blocks in zip(*iter_list):
    x = []
    for block in blocks[: -1]:
        x.append(block.pick_up)
        x.append(block.drop_off)
        if block.is_holiday:
            x.append(np.ones((n, n)))
        else:
            x.append(np.zeros((n, n)))

    x.append(np.zeros((n, n)) + blocks[-1].hour)
    if blocks[-1].is_holiday:
        x.append(np.ones((n, n)))
    else:
        x.append(np.zeros((n, n)))

    y = [blocks[-1].pick_up, blocks[-1].drop_off]

    xs.append(x)
    ys.append(y)

    print(len(ys))

xs = np.array(xs, dtype=np.int16)
ys = np.array(ys, dtype=np.int16)

print(xs.shape, ys.shape)

np.save("dataset_month%d/xs.npy" % args.month, xs)
np.save("dataset_month%d/ys.npy" % args.month, ys)