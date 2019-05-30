import numpy as np
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.misc
from collections import Counter
import random

#图像预处理的一些方法，自己封装一下

Path = r'C:\Users\Administrator\Desktop\2019\First\train_image3\001'
original_pic = []
for filename in os.listdir(Path):
    original_pic.append(Path+ "\\" + filename)

'''
#去除图像黑边
for i in range(len(original_pic)):
    a = ndimage.imread(original_pic[i])
    b = np.sum(a, axis=2)
    c = np.sum(b == 0)
    if c>100*100*0.3:
        os.remove(original_pic[i])'''
for i in range(len(original_pic)):

    #翻转图像，用于图像增强
    image_=ndimage.imread(original_pic[i])
    image_fliplr=np.fliplr(image_)
    scipy.misc.toimage(image_fliplr).save(original_pic[i]+ '_fliplr.jpg')

    '''
    image_ = ndimage.imread(original_pic[i])
    image_fliplr = np.rot90(image_, k=1)
    scipy.misc.toimage(image_fliplr).save(original_pic[i] + '_rot_' + str(1) + '.jpg')

    image_fliplr = np.rot90(image_, k=2)
    scipy.misc.toimage(image_fliplr).save(original_pic[i] + '_rot_' + str(2) + '.jpg')

    image_fliplr = np.rot90(image_, k=3)
    scipy.misc.toimage(image_fliplr).save(original_pic[i] + '_rot_' + str(3) + '.jpg')'''