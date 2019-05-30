import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import random
from collections import Counter
import math
import pickle

# 处理数据的模型

Train_Path = r'C:\Users\Administrator\Desktop\2019\First\train_visit\train_e'
Test_Path = r'C:\Users\Administrator\Desktop\2019\First\test_visit\teste'
file_001, file_002, file_003, file_004, file_005, file_006, file_007, file_008, file_009 = \
    [], [], [], [], [], [], [], [], []
file_test = []
BATCHSIZE = 500


# 数据置乱
def shuft(data):
    random.seed(2345)
    random.shuffle(data)


# 3层前馈神经网
def CNNop(input, keep_prob, training):
    with tf.variable_scope('cnnnet') as scope:
        input = tf.reshape(input, [-1, 493])
        w1 = tf.Variable(tf.random_normal([493, 886], stddev=1))
        out_1 = tf.matmul(input, w1)
        out_1 = tf.nn.dropout(out_1, keep_prob=keep_prob)
        bn_1 = tf.layers.batch_normalization(out_1, training=training)
        sig_1 = tf.nn.sigmoid(bn_1)

        w2 = tf.Variable(tf.random_normal([886, 1672], stddev=1))
        out_2 = tf.matmul(sig_1, w2)
        out_2 = tf.nn.dropout(out_2, keep_prob=keep_prob)
        bn_2 = tf.layers.batch_normalization(out_2, training=training)
        sig_2 = tf.nn.sigmoid(bn_2)

        w3 = tf.Variable(tf.random_normal([1672, 886], stddev=1))
        out_3 = tf.matmul(sig_2, w3)
        out_3 = tf.nn.dropout(out_3, keep_prob=keep_prob)
        bn_3 = tf.layers.batch_normalization(out_3, training=training)
        sig_3 = tf.nn.sigmoid(bn_3)

        w4 = tf.Variable(tf.random_normal([886, 9], stddev=1))
        bias = tf.Variable(tf.random_normal([9], mean=0.0, stddev=1), name="bias")
        y_ = tf.matmul(sig_3, w4) + bias
    return y_


input_x = tf.placeholder(tf.float32, shape=[None, 493])
input_y = tf.placeholder(tf.float32, shape=[None, 9])
# drop out
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)
y_ = CNNop(input_x, keep_prob, training)
# 结果为0123456789的百分比
y_prcent = tf.nn.softmax(y_)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=input_y))
global_step=tf.placeholder(tf.float32)
learning_rate=tf.train.exponential_decay(learning_rate=0.01,global_step=global_step,decay_steps=50000,decay_rate=0.9)
# 用于batchnormalization变量的更新
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prd = tf.equal(tf.argmax(y_, 1), tf.argmax(input_y, 1))
accuracyCover = tf.reduce_mean(tf.cast(correct_prd, tf.float32))
# True时进行训练，False进行预测
isTrain = False
with tf.Session() as sess:
    if isTrain:
        # 训练
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('./myModel/datamodel2')
        saver.restore(sess, ckpt)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuary', accuracyCover)
        writer = tf.summary.FileWriter('./my_graph/datamodel', sess.graph)
        summary_op = tf.summary.merge_all()
        for filename in os.listdir(Train_Path):
            if filename.endswith('_001.txt'):
                file_001.append(Train_Path + '\\' + filename)
            if filename.endswith('_002.txt'):
                file_002.append(Train_Path + '\\' + filename)
            if filename.endswith('_003.txt'):
                file_003.append(Train_Path + '\\' + filename)
            if filename.endswith('_004.txt'):
                file_004.append(Train_Path + '\\' + filename)
            if filename.endswith('_005.txt'):
                file_005.append(Train_Path + '\\' + filename)
            if filename.endswith('_006.txt'):
                file_006.append(Train_Path + '\\' + filename)
            if filename.endswith('_007.txt'):
                file_007.append(Train_Path + '\\' + filename)
            if filename.endswith('_008.txt'):
                file_008.append(Train_Path + '\\' + filename)
            if filename.endswith('_009.txt'):
                file_009.append(Train_Path + '\\' + filename)

        file = [file_001, file_002, file_003, file_004, file_005, file_006, file_007, file_008, file_009]
        # len =[9542,7538,3590,1358,3464,5507,3517,2617,2867]
        training_data = []
        training_target = []
        test_data = []
        test_target = []
        for k in range(9):
            w = file[k]
            for i in range(9542):
                tempi = i % len(file[k])
                with open(w[tempi]) as f:
                    l = f.readline().split('\t')[:-1]
                    target = np.zeros(shape=[9], dtype=np.float32)
                    target[k] = 1
                    training_data.append(list(map(float, l)))
                    training_target.append(target)
        shuft(training_data)
        shuft(training_target)
        # 9成数据用于训练，1成用于验证
        test_data = training_data[int(len(training_data) * 0.95):]
        test_target = training_target[int(len(training_target) * 0.95):]
        training_data = training_data[:int(len(training_data) * 0.95)]
        training_target = training_target[:int(len(training_target) * 0.95)]
        test_data_after = []
        test_target_after = []
        for i in range(len(test_data)):
            if test_data[i] not in test_data_after:
                test_data_after.append(test_data[i])
                test_target_after.append(test_target[i])

        for epoch in range(1000000):
            rand = np.random.randint(low=0, high=int(len(training_data) / BATCHSIZE))
            training_data_inp = training_data[rand * BATCHSIZE:(rand + 1) * BATCHSIZE]
            training_target_inp = training_target[rand * BATCHSIZE:(rand + 1) * BATCHSIZE]
            _, accuracyCover_p, loss_p, summary_op_p = sess.run([train_op, accuracyCover, loss, summary_op],
                                                                feed_dict={input_x: training_data_inp,
                                                                           input_y: training_target_inp,
                                                                           keep_prob: 0.5,
                                                                           training: True,global_step:epoch})
            print('train____epoch:%d,coorect:%f loss:%f' % (epoch, accuracyCover_p, loss_p))
            writer.add_summary(summary_op_p, epoch)
            if epoch % 100000 == 0:
                saver.save(sess, './myModel/datamodel2/model' + str(epoch / 100000) + '.cptk')
            if epoch % 1000 == 0:
                rand1 = np.random.randint(low=0, high=int(len(test_data_after) / BATCHSIZE))
                test_data_inp = test_data_after[rand1 * BATCHSIZE:(rand1 + 1) * BATCHSIZE]
                test_target_inp = test_target_after[rand1 * BATCHSIZE:(rand1 + 1) * BATCHSIZE]
                accuracyCover_p, loss_p, summary_op_k = sess.run([accuracyCover, loss, summary_op],
                                                                 feed_dict={input_x: test_data_inp,
                                                                            input_y: test_target_inp,
                                                                            keep_prob: 1.0,
                                                                            training: False})
                print('test____epoch:%d,coorect:%f loss:%f' % (epoch, accuracyCover_p, loss_p))
                # writer.add_summary(summary_op_k, epoch)

    else:
        # 加载模型进行预测
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('./myModel/datamodel2')
        saver.restore(sess, ckpt)
        AreaID = []
        CategoryID = []
        Percent_all = []
        for filename in os.listdir(Test_Path):
            file_test.append(Test_Path + '\\' + filename)

        file_test = np.asarray(file_test)

        for i in range(int(file_test.shape[0])):
            with open(file_test[i]) as f:
                l = f.readline().split('\t')[:-1]
                test_data_x = []
                test_data_x.append(list(map(float, l)))

            test_data_x = np.array(test_data_x)
            AreaID.append(file_test[i][-10:-4])
            y_p, y_prcent_p = sess.run([y_, y_prcent],
                                       feed_dict={input_x: test_data_x, keep_prob: 1.0, training: False})
            # print(y_prcent_p)
            test_result = np.argmax(y_p, axis=1)
            counter = Counter(test_result)
            # percent = []
            # for j in range(9):
            #    percent.append(counter[j] / test_result.shape[0])
            Percent_all.append(y_prcent_p[0])
            CategoryID.append(test_result[0] + 1)

        with open('result_data2.txt', 'w') as f:
            for j in range(file_test.shape[0]):
                f.write(str(AreaID[j]) + '\t00' + str(CategoryID[j]) + '\n')
        with open('percent4.txt', 'wb') as f:
            pickle.dump(Percent_all, f)
        # with open('percent5.txt', 'w') as f:
        # for j in range(file_test.shape[0]):
        #    f.write(str(Percent_all[j])+'\n')
