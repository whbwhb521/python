# encoding=utf-8
import numpy as np
import cv2
import time
import pickle

#用于结合data和picture的结果

# percent1.txt保存的是用图像训练出来的结果的百分比
# percent2.txt保存的是用数据训练出来的结果的百分比
with open('percent3.txt', 'rb') as f:
    Percent1 = pickle.load(f)
with open('percent4.txt', 'rb') as f:
    Percent2 = pickle.load(f)
with open('AreaID.txt', 'rb') as f:
    AreaID = pickle.load(f)

# 相乘算出总体的预测结果
Percent3 = np.multiply(Percent1, Percent2)
test_result = np.argmax(Percent3, axis=1)

# 将结果写入txt文档中
with open('result_data4.txt', 'w') as f:
    for i in range(len(test_result)):
        f.write(AreaID[i] + '\t00' + str(test_result[i]+1) + '\n')
