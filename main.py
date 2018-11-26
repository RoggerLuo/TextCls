import tensorflow as tf
import numpy as np
from model import getTrainingModel,getPredictionModel
from getLoss import getLoss
from embed import str2embed
from os.path import exists
from tensorflow.contrib import layers

def optimizer(loss):    
    return layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.001 #0.001
    )


def test(shape=[200,200], stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        return tf.get_variable('test',initializer=initial)

def prediction():
    embedingPlaceholder,y = getPredictionModel()
    def predict_feed_fn(string):
        x = str2embed(string)
        return {embedingPlaceholder:x}

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # 每次不写就会报错

    if exists('./ckpt'):
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)

    string = '这个是圆圆的分类'
    _y = sess.run(y,feed_dict=predict_feed_fn(string))
    print(_y)
    string = '这个是圆的分类'
    _y = sess.run(y,feed_dict=predict_feed_fn(string))
    print(_y)

def train():

    embedingPlaceholder,y = getTrainingModel()
    y_labelPlaceholder,cross_entropy,accuracy = getLoss(y)
    train_op = optimizer(cross_entropy)

    # tes = test()
    def feed_fn(string,flag):
        label = [0]
        if flag == True:
            label = [1]
        # for string in articleList:
        x = str2embed(string)
        return {embedingPlaceholder:x,y_labelPlaceholder:label}



    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # 每次不写就会报错

    if exists('./ckpt'):
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)

    for i in range(200):
        string = '这个是圆的分类'
        print(string)
        flag = False
    # print(sess.run(tes))

        loss,acc,_ = sess.run([cross_entropy,accuracy,train_op],feed_dict=feed_fn(string,flag))
        

        print('----[acc]----')
        print(acc)
        print('loss:',loss)
        print('----------------------------------------')

        print('这个是圆圆的分类')
        string = '这个是圆圆的分类'
        flag = True
    # print(sess.run(tes))

        loss,acc,_ = sess.run([cross_entropy,accuracy,train_op],feed_dict=feed_fn(string,flag))
        

        print('----[acc]----')
        print(acc)
        print('loss:',loss)
        print('----------------------------------------')

    # if i%100 == 0:
    saver.save(sess, 'ckpt/model.ckpt')


# train()

prediction()

