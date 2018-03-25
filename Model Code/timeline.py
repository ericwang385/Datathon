from hoursblock import HourBlock
import datetime
import bisect
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Data set text file")
parser.add_argument('--year', type=int, default = 2013, help='Data set year, default 2013')
parser.add_argument('--month', type=int, default = 1, help='Data set month, default 1')
parser.add_argument('--timedelta', type=int, default = 1,
                    help='Timedelta, or known as time frame length, in hour, default 1h')
parser.add_argument('--total_index', type=int, default = 15,
                    help='Number of fraction to cut, best the cup core number, default 15')
parser.add_argument('--fraction_index', type=int, default = 15,
                    help='Number of fraction to run')
args = parser.parse_args()

total_index = args.total_index
fraction_index = args.fraction_index
year = args.year
month = args.month
timedelta = args.timedelta

start = datetime.datetime(year, month, 1, 0)
current = datetime.datetime(year, month, 1, 0)
if month == 12:
    end = datetime.datetime(year + 1, 1, 1, 0)
else:
    end = datetime.datetime(year, month + 1, 1, 0)
delta = datetime.timedelta(hours=timedelta)

time_line = []

while current < end:
    time_line.append(HourBlock(current))
    current += delta

def record(is_pick_up, time, passenger_count, lon, lat):
    index = bisect.bisect_left(time_line, HourBlock(time))
    try:
        time_line[index].record(is_pick_up, passenger_count, lon, lat)
    except Exception as e:
        print(e)

f = open(str(month) + ".csv", "r")
f.readline()  # get rid of the head
line_count = -1
for line_count, _ in enumerate(f):
    pass
line_count += 1
print(line_count)

f = open(str(month) + ".csv", "r")
f.readline()  # get rid of the head

"""Unnamed: 0,pickup_datetime,dropoff_datetime, 2
passenger_count,trip_time_in_secs,pickup_longitude, 5
pickup_latitude,dropoff_longitude,dropoff_latitude 8"""

i = 1
start = time.time()
one_line = f.readline()

while i < (line_count // total_index) * (fraction_index - 1):
    f.readline()
    i += 1

starting_id = i
ending_id = (line_count // total_index) * fraction_index
print(starting_id, ending_id)
while one_line:
    items = one_line.split(',')

    try:
        pickup_time = datetime.datetime.strptime(items[0], "%Y-%m-%d %H:%M:%S")
        record(True, pickup_time, int(items[2]), float(items[4]), float(items[5]))

        drop_off_time = datetime.datetime.strptime(items[1], "%Y-%m-%d %H:%M:%S")
        record(False, drop_off_time, int(items[2]), float(items[6]), float(items[7]))
    except Exception as e:
        print(e)

    one_line = f.readline()
    i += 1

    if i % 10000 == 0:
        print(time.time() - start, i, fraction_index)

    if i > (line_count // total_index) * fraction_index:
        break

import pickle
pickle.dump(time_line, open("dataset_month%d/%d_%d.pkl" % (month, month, fraction_index), "wb+"))