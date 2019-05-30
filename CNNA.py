import tensorflow as tf
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle

# 图像效果较差，需要大改

Train_Path = r'C:\Users\Administrator\Desktop\2019\First\train_image3'
Test_Path = r'C:\Users\Administrator\Desktop\2019\First\test_image\test'

# kv核高通滤波
Kkv = np.zeros([3, 3, 3, 3], dtype=np.float32)
Kkv[:, :, 0, 0] = np.array([[[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]]
                            ], dtype=np.float32) / (8 * 255)
Kkv[:, :, 1, 0] = np.array([[[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]]
                            ], dtype=np.float32) / (8 * 255)
Kkv[:, :, 2, 0] = np.array([[[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]]
                            ], dtype=np.float32) / (8 * 255)
Kkv01 = tf.Variable(Kkv, name='Kkv01')


# Test集的图像路径
def get_Test_filename(file_dir):
    original_pic = []
    for filename in os.listdir(file_dir):
        original_pic.append(file_dir + "\\" + filename)

    return np.asarray(original_pic)


'''
# 图像路径和标签
def get_filename(file_dir):
    original_pic = []
    labe1_pic = []
    ffile = os.listdir(file_dir)
    for k in range(len(ffile)):
        ls = file_dir + "\\" + ffile[k]
        file = os.listdir(ls)
        for i in range(9449):
            tempi = i % len(file)
            original_pic.append(ls + '\\' + file[tempi])
            labe1_pic.append(k)

    return np.asarray(original_pic), np.asarray(labe1_pic)'''


# 图像路径和标签
def get_filename(file_dir):
    original_pic = []
    labe1_pic = []
    for filename in os.listdir(file_dir):
        ls = file_dir + "\\" + filename
        for subfilename in os.listdir(ls):
            original_pic.append(ls + '\\' + subfilename)
            labe1_pic.append(filename)

    return np.asarray(original_pic), np.asarray(labe1_pic)


# 图像置乱
def shuffle_image(image, label):
    random.seed(234)
    random.shuffle(image)
    random.seed(234)
    random.shuffle(label)
    return image, label


def sub_block(input, name, istraining, k):
    with tf.variable_scope(name) as scope:
        bn_0 = tf.layers.batch_normalization(input, training=istraining)
        rl_0 = tf.nn.relu(bn_0)
        kerne_0 = tf.Variable(tf.random_normal([1, 1, k, int(k / 2)], mean=0.0, stddev=0.01),
                              name="kerne_0")  # [height,width,input,output]
        con_0 = tf.nn.conv2d(rl_0, filter=kerne_0, strides=[1, 1, 1, 1], padding='SAME')

        bn_1 = tf.layers.batch_normalization(con_0, training=istraining)
        rl_1 = tf.nn.relu(bn_1)
        kerne_1 = tf.Variable(tf.random_normal([3, 3, int(k / 2), 32], mean=0.0, stddev=0.01),
                              name="kerne_1")  # [height,width,input,output]
        con_1 = tf.nn.conv2d(rl_1, filter=kerne_1, strides=[1, 1, 1, 1], padding='SAME')
        return con_1


def block_0(input, name, istraining, k0, k1):
    with tf.variable_scope(name) as scope:
        sb1 = sub_block(input, name='sb1', istraining=istraining, k=k0)
        sb2 = sub_block(sb1, name='sb2', istraining=istraining, k=k1)
        sb3_input = tf.concat([sb1, sb2], axis=-1)
        sb3 = sub_block(sb3_input, name='sb3', istraining=istraining, k=2 * k1)
        sb4_input = tf.concat([sb3_input, sb3], axis=-1)
        sb4 = sub_block(sb4_input, name='sb4', istraining=istraining, k=3 * k1)
        sb5_input = tf.concat([sb4_input, sb4], axis=-1)
        sb5 = sub_block(sb5_input, name='sb5', istraining=istraining, k=4 * k1)
        sb6_input = tf.concat([sb5_input, sb5], axis=-1)
        sb6 = sub_block(sb6_input, name='sb6', istraining=istraining, k=5 * k1)
        op = tf.concat([sb6_input, sb6], axis=-1)
        return op


def block_1(input, name, istraining, k0, k1):
    with tf.variable_scope(name) as scope:
        sb1 = sub_block(input, name='sb1', istraining=istraining, k=k0)
        sb2 = sub_block(sb1, name='sb2', istraining=istraining, k=k1)
        sb3_input = tf.concat([sb1, sb2], axis=-1)
        sb3 = sub_block(sb3_input, name='sb3', istraining=istraining, k=2 * k1)
        sb4_input = tf.concat([sb3_input, sb3], axis=-1)
        sb4 = sub_block(sb4_input, name='sb4', istraining=istraining, k=3 * k1)
        sb5_input = tf.concat([sb4_input, sb4], axis=-1)
        sb5 = sub_block(sb5_input, name='sb5', istraining=istraining, k=4 * k1)
        sb6_input = tf.concat([sb5_input, sb5], axis=-1)
        sb6 = sub_block(sb6_input, name='sb6', istraining=istraining, k=5 * k1)
        sb7_input = tf.concat([sb6_input, sb6], axis=-1)
        sb7 = sub_block(sb7_input, name='sb7', istraining=istraining, k=6 * k1)
        sb8_input = tf.concat([sb7_input, sb7], axis=-1)
        sb8 = sub_block(sb8_input, name='sb8', istraining=istraining, k=7 * k1)
        sb9_input = tf.concat([sb8_input, sb8], axis=-1)
        sb9 = sub_block(sb9_input, name='sb9', istraining=istraining, k=8 * k1)
        sb10_input = tf.concat([sb9_input, sb9], axis=-1)
        sb10 = sub_block(sb10_input, name='sb10', istraining=istraining, k=9 * k1)
        sb11_input = tf.concat([sb10_input, sb10], axis=-1)
        sb11 = sub_block(sb11_input, name='sb11', istraining=istraining, k=10 * k1)
        sb12_input = tf.concat([sb11_input, sb11], axis=-1)
        sb12 = sub_block(sb12_input, name='sb12', istraining=istraining, k=11 * k1)
        op = tf.concat([sb12_input, sb12], axis=-1)
        return op


def block_2(input, name, istraining, k0, k1):
    with tf.variable_scope(name) as scope:
        sb1 = sub_block(input, name='sb1', istraining=istraining, k=k0)
        sb2 = sub_block(sb1, name='sb2', istraining=istraining, k=k1)
        sb3_input = tf.concat([sb1, sb2], axis=-1)
        sb3 = sub_block(sb3_input, name='sb3', istraining=istraining, k=2 * k1)
        sb4_input = tf.concat([sb3_input, sb3], axis=-1)
        sb4 = sub_block(sb4_input, name='sb4', istraining=istraining, k=3 * k1)
        sb5_input = tf.concat([sb4_input, sb4], axis=-1)
        sb5 = sub_block(sb5_input, name='sb5', istraining=istraining, k=4 * k1)
        sb6_input = tf.concat([sb5_input, sb5], axis=-1)
        sb6 = sub_block(sb6_input, name='sb6', istraining=istraining, k=5 * k1)
        sb7_input = tf.concat([sb6_input, sb6], axis=-1)
        sb7 = sub_block(sb7_input, name='sb7', istraining=istraining, k=6 * k1)
        sb8_input = tf.concat([sb7_input, sb7], axis=-1)
        sb8 = sub_block(sb8_input, name='sb8', istraining=istraining, k=7 * k1)
        sb9_input = tf.concat([sb8_input, sb8], axis=-1)
        sb9 = sub_block(sb9_input, name='sb9', istraining=istraining, k=8 * k1)
        sb10_input = tf.concat([sb9_input, sb9], axis=-1)
        sb10 = sub_block(sb10_input, name='sb10', istraining=istraining, k=9 * k1)
        sb11_input = tf.concat([sb10_input, sb10], axis=-1)
        sb11 = sub_block(sb11_input, name='sb11', istraining=istraining, k=10 * k1)
        sb12_input = tf.concat([sb11_input, sb11], axis=-1)
        sb12 = sub_block(sb12_input, name='sb12', istraining=istraining, k=11 * k1)
        sb13_input = tf.concat([sb12_input, sb12], axis=-1)
        sb13 = sub_block(sb13_input, name='sb13', istraining=istraining, k=12 * k1)
        sb14_input = tf.concat([sb13_input, sb13], axis=-1)
        sb14 = sub_block(sb14_input, name='sb14', istraining=istraining, k=13 * k1)
        sb15_input = tf.concat([sb14_input, sb14], axis=-1)
        sb15 = sub_block(sb15_input, name='sb15', istraining=istraining, k=14 * k1)
        sb16_input = tf.concat([sb15_input, sb15], axis=-1)
        sb16 = sub_block(sb16_input, name='sb16', istraining=istraining, k=15 * k1)
        op = tf.concat([sb16_input, sb16], axis=-1)
        return op


# 随便写的CNN网络
def CNNop(input,keep_prob, istraining):
    with tf.variable_scope('cnnnet') as scope:
        input = tf.reshape(input, [-1, 100, 100, 3])
        out = tf.nn.conv2d(input, filter=Kkv01, strides=[1, 1, 1, 1], padding='SAME')
        k_before = tf.Variable(tf.random_normal([7, 7, 3, 32], mean=0.0, stddev=0.01), name="k_before")
        # size =50*50
        conv_before = tf.nn.conv2d(out, filter=k_before, strides=[1, 2, 2, 1], padding='SAME')
        # size=25*25
        pool_before = tf.nn.max_pool(conv_before, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     name='pool_before')
        b0 = block_0(pool_before, 'b0', istraining, k0=32, k1=32)
        # 中间层0  op=192
        bn0 = tf.layers.batch_normalization(b0, training=istraining)
        k0 = tf.Variable(tf.random_normal([1, 1, 192, 96], mean=0.0, stddev=0.01), name="k0")
        conv0 = tf.nn.conv2d(bn0, filter=k0, strides=[1, 1, 1, 1], padding='SAME')
        pool0 = tf.nn.avg_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
        #
        # size=13*13    96

        b1 = block_1(pool0, 'b1', istraining, k0=96, k1=32)
        # 中间层1   op=32*12  384
        bn1 = tf.layers.batch_normalization(b1, training=istraining)
        k1 = tf.Variable(tf.random_normal([1, 1, 384, 192], mean=0.0, stddev=0.01), name="k1")
        conv1 = tf.nn.conv2d(bn1, filter=k1, strides=[1, 1, 1, 1], padding='SAME')
        pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #
        # size=7*7  32*16  512
        b2 = block_2(pool1, 'b2', istraining, k0=192, k1=32)
        '''
        # 中间层2
        bn2 = tf.layers.batch_normalization(b2, training=istraining)
        k2 = tf.Variable(tf.random_normal([1, 1, 3, 64], mean=0.0, stddev=0.01), name="k2")
        conv2 = tf.nn.conv2d(bn2, filter=k2, strides=[1, 1, 1, 1], padding='SAME')
        pool2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')
        # 
        b3 = block(pool2, 'b3', istraining)
        '''
        bn4 = tf.layers.batch_normalization(b2, training=istraining)
        pool_global = tf.nn.avg_pool(bn4, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME', name='pool_g2')

        pool_reshape = tf.reshape(pool_global, [-1, 512])
        # print(pool_reshape)
        weights = tf.Variable(tf.random_normal([512, 9], mean=0.0, stddev=0.01), name="weights")
        #bias = tf.Variable(tf.random_normal([9], mean=0.0, stddev=0.01), name="bias")
        y_ = tf.matmul(pool_reshape, weights)
        y_ = tf.nn.dropout(y_, keep_prob=keep_prob)
    return y_


# 图像路径和标签
a, b = get_filename(Train_Path)
c = get_Test_filename(Test_Path)
shuffle_image(a, b)

# 训练验证比
ratio = 0.9
s = np.int(b.shape[0] * ratio)
s0 = b.shape[0] - s
a_train = a[:s]
b_train = b[:s]
a_val = a[s:]
b_val = b[s:]

BATCHSIZE = 100
data_x = np.zeros([BATCHSIZE, 100, 100, 3])
data_y = np.zeros([BATCHSIZE, 9])
input_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
keep_prob= tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, shape=[None, 9])
istraining = tf.placeholder(tf.bool)
y_ = CNNop(input_image,keep_prob, istraining)
y_prcent = tf.nn.softmax(y_)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
correct_prd = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracyCover = tf.reduce_mean(tf.cast(correct_prd, tf.float32))

isTrain = False

with tf.Session() as sess:
    if isTrain:
        # 训练
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuary', accuracyCover)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('./myModel')
        saver.restore(sess, ckpt)
        writer = tf.summary.FileWriter('./my_graph', sess.graph)
        summary_op = tf.summary.merge_all()

        for epoch in range(1000000):
            randnum = epoch % int(s / BATCHSIZE)
            image_train = a_train[randnum * BATCHSIZE:(randnum + 1) * BATCHSIZE]
            label_train = b_train[randnum * BATCHSIZE:(randnum + 1) * BATCHSIZE]
            for j in range(BATCHSIZE):
                imc = ndimage.imread(image_train[j])
                data_x[j, :, :, :] = imc
                data_y[j, :] = [(label_train[j] == var) for var in
                                ['001', '002', '003', '004', '005', '006', '007', '008', '009']]
                # print(data_y[j,:])

            _, accuracyCover_p, summary_op_p, loss_p, y_p = sess.run([train_op, accuracyCover, summary_op, loss, y_],
                                                                     feed_dict={input_image: data_x, y: data_y,
                                                                                istraining: True,keep_prob:0.5})
            print('epoch:%d,coorect:%f,loss:%f' % (epoch, accuracyCover_p, loss_p))

            if epoch % 1000 == 0:
                randnum = np.random.randint(low=0, high=int(s0 / BATCHSIZE))
                image_val = a_val[randnum * BATCHSIZE:(randnum + 1) * BATCHSIZE]
                label_val = b_val[randnum * BATCHSIZE:(randnum + 1) * BATCHSIZE]
                for k in range(BATCHSIZE):
                    imc = ndimage.imread(image_val[k])
                    data_x[k, :, :, :] = imc
                    data_y[k, :] = [(label_val[k] == var) for var in
                                    ['001', '002', '003', '004', '005', '006', '007', '008', '009']]
                accuracyCover_p = sess.run(accuracyCover,
                                           feed_dict={input_image: data_x, y: data_y, istraining: False,keep_prob:1.0})
                print('val------------epoch:%d,coorect:%f' % (epoch, accuracyCover_p))

            if epoch % 10 == 0:
                writer.add_summary(summary_op_p, epoch)

            if epoch % 10000 == 0:
                saver.save(sess, './myModel/ASDLmodel' + str(epoch / 10000) + '.cptk')


    else:
        # 测试
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('./myModel')
        saver.restore(sess, ckpt)
        AreaID = []
        CategoryID = []
        Percent_all = []
        for i in range(int(c.shape[0] / BATCHSIZE)):
            image_test = c[i * BATCHSIZE:(i + 1) * BATCHSIZE]
            for j in range(BATCHSIZE):
                imc = ndimage.imread(image_test[j])
                data_x[j, :, :, :] = imc
                AreaID.append(image_test[j][-10:-4])

            y_p, y_prcent_p = sess.run([y_, y_prcent], feed_dict={input_image: data_x, istraining: False,keep_prob:1.0})
            test_result = np.argmax(y_p, axis=1)
            for j in range(BATCHSIZE):
                CategoryID.append(test_result[j] + 1)
                Percent_all.append(y_prcent_p[j])

        with open('result_data3.txt', 'w') as f:
            for j in range(c.shape[0]):
                f.write(str(AreaID[j]) + '\t00' + str(CategoryID[j]) + '\n')
        with open('percent3.txt', 'wb') as f:
            pickle.dump(Percent_all, f)
        # with open('AreaID.txt', 'wb') as f:
        #    pickle.dump(AreaID,f)
