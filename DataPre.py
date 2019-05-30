import os
import re
import numpy as np
from collections import Counter

# 用于数据预处理
Path = r'C:\Users\Administrator\Desktop\2019\First\train_visit\train'


# Y:年份后两位 M：月（大于等于3，小于等于14） D：日 C：年份前2位
# W=Y+Y//4+C//4-2*C+26*(M+1)//10+D-1
# 年转周
def year2week(str_day):
    str_week = []
    for i in range(len(str_day)):
        C = int(str_day[i][0:2])
        Y = int(str_day[i][2:4])
        D = int(str_day[i][6:8])
        if int(str_day[i][4:6]) < 3:
            M = int(str_day[i][4:6]) + 12
        else:
            M = int(str_day[i][4:6])

        W = (Y + Y // 4 + C // 4 - 2 * C + 26 * (M + 1) // 10 + D - 1) % 7
        str_week.append(W)
    return str_week


# 时间转成2值形式
def hour2YN(str_hour):
    str_YN = []
    for i in range(len(str_hour)):
        a = np.zeros(shape=[24], dtype=np.int32)
        for j in range(24):
            if j < 10:
                k = '0' + str(j)
            else:
                k = str(j)
            if k in str_hour[i]:
                a[j] = 1
        str_YN.append(a)
    return str_YN


# 自定义的除法，防止除0用
def delet(a, b):
    if b == 0:
        result = 0
    else:
        result = a / b
    return result


def delet2(a, b):
    if 0 in b:
        b = b + 0.00000000001
    return a / b


xkk = os.listdir(Path)
for filename in range(len(xkk)):
    road = Path + "\\" + xkk[filename]
    str_day = []
    str_hour = []
    with open(road) as f:
        stri = f.read()
        str_sp = stri.split()
        for i in range(1, len(str_sp), 2):
            data = re.split('&|,', str_sp[i])
            for j in range(0, len(data), 2):
                str_day.append(data[j])
                str_hour.append(data[j + 1])

# str_week = year2week(str_day)
# counter = Counter(str_week)
# percent = []

    hoilday=[['20181001', '20181002', '20181003', '20181004', '20181005', '20181006', '20181007'],
             ['20181230', '20181231', '20190101'],
             ['20190204', '20190205', '20190206', '20190207', '20190208', '20190209', '20190210']]
    str_YN = hour2YN(str_hour)
    str_start = np.argmax(str_YN, axis=1)

    for round in range(3):
        str_hoilday = []
        sum_hoilday = 0
        for i in range(len(str_day)):
            if str_day[i] in hoilday[round]:
                str_hoilday.append(i)

        percent_hoilday = len(str_hoilday) / len(str_day)
        # for i in range(7):
        #    percent.append(counter[i] / len(str_week))
        for i in range(len(str_hoilday)):
            sum_hoilday = sum_hoilday + np.mean(str_YN[int(str_hoilday[i])])
        if len(str_hoilday) == 0:
            avg_hour_hoilday = 0
        else:
            avg_hour_hoilday = sum_hoilday / len(str_hoilday)
        str_hoilday_start_time = np.zeros(shape=[24], dtype=np.float32)
        for i in str_hoilday:
            str_hoilday_start_time[str_start[i]] = str_hoilday_start_time[str_start[i]] + 1

        if np.sum(str_hoilday_start_time) == 0:
            str_hoilday_start_time = np.zeros(shape=[24], dtype=np.float32)
        else:
            str_hoilday_start_time = str_hoilday_start_time / np.sum(str_hoilday_start_time)
        kxtime_hoilday = np.zeros(shape=[23])
        for i in str_hoilday:
            l = np.argwhere(str_YN[i] == 1)
            if len(l) > 1:
                max = 1
                for j in range(len(l) - 1):
                    if l[j + 1][0] - l[j][0] > 1:
                        if (l[j + 1][0] - l[j][0]) > max:
                            max = l[j + 1][0] - l[j][0]
                kxtime_hoilday[max - 1] = kxtime_hoilday[max - 1] + 1
            else:
                kxtime_hoilday[0] = kxtime_hoilday[0] + 1
        sum = np.sum(kxtime_hoilday)
        if sum == 0:
            kxtime_hoilday = np.zeros([23])
        else:
            kxtime_hoilday = kxtime_hoilday / np.sum(kxtime_hoilday)
        road_a = r'C:\Users\Administrator\Desktop\2019\First\train_visit\train_e' + '\\' + xkk[filename]
        with open(road_a, 'a') as f:
            f.write(
                str(percent_hoilday) + '\t' + str(avg_hour_hoilday) + '\t' + str(str_hoilday_start_time[0]) + '\t' + str(
                    str_hoilday_start_time[1]) + '\t' + str(str_hoilday_start_time[2]) + '\t' + str(
                    str_hoilday_start_time[3]) + '\t' + str(str_hoilday_start_time[4]) + '\t' + str(
                    str_hoilday_start_time[5]) + '\t' + str(str_hoilday_start_time[6]) + '\t' + str(
                    str_hoilday_start_time[7]) + '\t' + str(str_hoilday_start_time[8]) + '\t' + str(
                    str_hoilday_start_time[9]) + '\t' + str(str_hoilday_start_time[10]) + '\t' + str(
                    str_hoilday_start_time[11]) + '\t' + str(str_hoilday_start_time[12]) + '\t' + str(
                    str_hoilday_start_time[13]) + '\t' + str(str_hoilday_start_time[14]) + '\t' + str(
                    str_hoilday_start_time[15]) + '\t' + str(str_hoilday_start_time[16]) + '\t' + str(
                    str_hoilday_start_time[17]) + '\t' + str(str_hoilday_start_time[18]) + '\t' + str(
                    str_hoilday_start_time[19]) + '\t' + str(str_hoilday_start_time[20]) + '\t' + str(
                    str_hoilday_start_time[21]) + '\t' + str(str_hoilday_start_time[22]) + '\t' + str(
                    str_hoilday_start_time[23]) + '\t' + str(kxtime_hoilday[0]) + '\t' + str(
                    kxtime_hoilday[1]) + '\t' + str(
                    kxtime_hoilday[2]) + '\t' + str(kxtime_hoilday[3]) + '\t' + str(kxtime_hoilday[4]) + '\t' + str(
                    kxtime_hoilday[5]) + '\t' + str(kxtime_hoilday[6]) + '\t' + str(kxtime_hoilday[7]) + '\t' + str(
                    kxtime_hoilday[8]) + '\t' + str(kxtime_hoilday[9]) + '\t' + str(kxtime_hoilday[10]) + '\t' + str(
                    kxtime_hoilday[11]) + '\t' + str(kxtime_hoilday[12]) + '\t' + str(kxtime_hoilday[13]) + '\t' + str(
                    kxtime_hoilday[14]) + '\t' + str(kxtime_hoilday[15]) + '\t' + str(kxtime_hoilday[16]) + '\t' + str(
                    kxtime_hoilday[17]) + '\t' + str(kxtime_hoilday[18]) + '\t' + str(kxtime_hoilday[19]) + '\t' + str(
                    kxtime_hoilday[20]) + '\t' + str(kxtime_hoilday[21]) + '\t' + str(kxtime_hoilday[22]) + '\t' + str(
                    kxtime_hoilday[22]) + '\t')
'''
kxtime = np.zeros(shape=[7, 23])
for i in range(len(str_YN)):
    l = np.argwhere(str_YN[i] == 1)
    if len(l) > 1:
        max = 1
        for j in range(len(l) - 1):
            if l[j + 1][0] - l[j][0] > 1:
                if (l[j + 1][0] - l[j][0]) > max:
                    max = l[j + 1][0] - l[j][0]
        kxtime[str_week[i]][max - 1] = kxtime[str_week[i]][max - 1] + 1
    else:
        kxtime[str_week[i]][0] = kxtime[str_week[i]][0] + 1

for i in range(7):
    sum = np.sum(kxtime, axis=1)[i]
    if sum == 0:
        kxtime[i] = np.zeros([23])
    else:
        kxtime[i] = kxtime[i] / np.sum(kxtime, axis=1)[i]

# 周一到周七每天呆多久
avg_hour = np.zeros(shape=[7], dtype=np.float32)
for i in range(len(str_week)):
    avg_hour[str_week[i]] = avg_hour[str_week[i]] + np.mean(str_YN[i])
for i in range(7):
    avg_hour[i] = delet(avg_hour[i], counter[i])

# 周一到周七中起始时间在24小时中的占比
str_week_start_time = np.zeros(shape=[7, 24], dtype=np.float32)
for i in range(len(str_start)):
    str_week_start_time[str_week[i]][str_start[i]] = str_week_start_time[str_week[i]][str_start[i]] + 1
str_week_start_time = np.transpose(delet2(np.transpose(str_week_start_time), np.sum(str_week_start_time, axis=1)))

# 写入txt文件
#

with open('test.txt', 'w') as f:
    for i in range(7):
        f.write(
            str(percent[i]) + '\t' + str(avg_hour[i]) + '\t' + str(str_week_start_time[i][0]) + '\t' + str(
                str_week_start_time[i][1]) + '\t' + str(str_week_start_time[i][2]) + '\t' + str(
                str_week_start_time[i][3]) + '\t' + str(str_week_start_time[i][4]) + '\t' + str(
                str_week_start_time[i][5]) + '\t' + str(str_week_start_time[i][6]) + '\t' + str(
                str_week_start_time[i][7]) + '\t' + str(str_week_start_time[i][8]) + '\t' + str(
                str_week_start_time[i][9]) + '\t' + str(str_week_start_time[i][10]) + '\t' + str(
                str_week_start_time[i][11]) + '\t' + str(str_week_start_time[i][12]) + '\t' + str(
                str_week_start_time[i][13]) + '\t' + str(str_week_start_time[i][14]) + '\t' + str(
                str_week_start_time[i][15]) + '\t' + str(str_week_start_time[i][16]) + '\t' + str(
                str_week_start_time[i][17]) + '\t' + str(str_week_start_time[i][18]) + '\t' + str(
                str_week_start_time[i][19]) + '\t' + str(str_week_start_time[i][20]) + '\t' + str(
                str_week_start_time[i][21]) + '\t' + str(str_week_start_time[i][22]) + '\t' + str(
                str_week_start_time[i][23]) + '\t' + str(kxtime[i][0]) + '\t' + str(kxtime[i][1]) + '\t' + str(
                kxtime[i][2]) + '\t' + str(kxtime[i][3]) + '\t' + str(kxtime[i][4]) + '\t' + str(
                kxtime[i][5]) + '\t' + str(
                kxtime[i][6]) + '\t' + str(kxtime[i][7]) + '\t' + str(kxtime[i][8]) + '\t' + str(
                kxtime[i][9]) + '\t' + str(
                kxtime[i][10]) + '\t' + str(kxtime[i][11]) + '\t' + str(kxtime[i][12]) + '\t' + str(
                kxtime[i][13]) + '\t' + str(
                kxtime[i][14]) + '\t' + str(kxtime[i][15]) + '\t' + str(kxtime[i][16]) + '\t' + str(
                kxtime[i][17]) + '\t' + str(
                kxtime[i][18]) + '\t' + str(kxtime[i][19]) + '\t' + str(kxtime[i][20]) + '\t' + str(
                kxtime[i][21]) + '\t' + str(
                kxtime[i][22]) + '\t')'''
