import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("month", type=int, help="Data set month")
parser.add_argument("--total_fraction", default=15, type=int, help="Total fraction, default 15")
args = parser.parse_args()

file_name = "dataset_month%d/%d_%d.pkl"

i = 0
total = pickle.load(open(file_name % (args.month, args.month, i + 1), "rb"))

for i in range(1, args.total_fraction):
    current = pickle.load(open(file_name % (args.month, args.month, i + 1), "rb"))
    for j in range(len(current)):
        print(i, j)
        one = current[j]
        total[j] += current[j]

pickle.dump(total, open("dataset_month%d/%d.pkl" % (args.month, args.month), "wb+"))