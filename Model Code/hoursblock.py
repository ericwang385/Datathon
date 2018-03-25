import numpy as np
import datetime
import math
import holidays

SPLITTING_COUNT = 200


class HourBlock(object):
    """Hour block"""

    def __init__(self, date_hour):
        self.date_hour = self.init_date_hour(date_hour)
        self.is_holiday = self.init_is_holiday(date_hour)

        n = SPLITTING_COUNT
        self.pick_up = np.zeros((n, n))
        self.drop_off = np.zeros((n, n))

    def __lt__(self, hour_block):
        return self.date_hour < hour_block.date_hour

    @property
    def hour(self):
        return self.date_hour.hour

    def init_date_hour(self, date_hour):
        year = date_hour.year
        month = date_hour.month
        day = date_hour.day
        hour = date_hour.hour
        return datetime.datetime(year, month, day, hour)

    def init_is_holiday(self, date_hour):
        us_holidays = holidays.UnitedStates()
        if date_hour.weekday() == 5 or 6:
            return True
        elif datetime in us_holidays:
            return True
        else:
            return False

    def record(self, is_pick_up, passenger_count, lon, lat):
        if is_pick_up:
            op_matrix = self.pick_up
        else:
            op_matrix = self.drop_off

        i, j, legal = self.geo_to_sub(lon, lat)
        if legal:
            # print(passenger_count, lon, lat, self.date_hour)
            op_matrix[i][j] += passenger_count

    def __iadd__(self, other):
        if self.date_hour != other.date_hour:
            raise ValueError("Time is not matched!")
        self.pick_up += other.pick_up
        self.drop_off += other.drop_off
        return self

    def geo_to_sub(self, lon, lat):
        x = lon
        y = lat
        n = SPLITTING_COUNT

        maxla = 40.915430
        minla = 40.495889
        maxlo = -73.699973
        minlo = -74.255717

        a = (maxla - minla) / n
        b = (maxlo - minlo) / n

        x = math.floor((x - minlo) / b)
        y = n - 1 - math.floor((y - minla) / a)

        c = x
        x = y
        y = c
        return x, y, x >= 0 and x <= n - 1 and y >= 0 and y <= n - 1


if __name__ == "__main__":
    h = HourBlock(datetime.datetime(2013, 1, 1, 2, 2 ,2))
    print(h.date_hour)
