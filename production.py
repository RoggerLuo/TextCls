import tensorflow as tf
import numpy as np
from model import getTrainingModel,getPredictionModel
from getLoss import getLoss
from embed import str2embed
from os.path import exists,join
from tensorflow.contrib import layers
from random import choice

def optimizer(loss):    
    return layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.001
    )

def prediction(entryList,categoryId):
    embedingPlaceholder,y = getPredictionModel()
    def predict_feed_fn(string):
        x = str2embed(string)
        return {embedingPlaceholder:x}

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # 每次不写就会报错
    path = join('ckpt',categoryId)
    if exists(path):
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)

    returnList = []
    for entry in entryList:
        content = entry[2]
        note_id = entry[0]
        _y = sess.run(y,feed_dict=predict_feed_fn(content))
        print(_y)
        score = _y[0][1]
        if score > 0.5:            
            returnList.append({'score':score,'note_id':note_id})
    return returnList

def train(categoryId,entryListYes,entryListNo):
    embedingPlaceholder,y = getTrainingModel()
    y_labelPlaceholder,cross_entropy,accuracy = getLoss(y)
    train_op = optimizer(cross_entropy)
    def feed_fn(string,flag):
        label = [0]
        if flag == True:
            label = [1]
        x = str2embed(string)
        return {embedingPlaceholder:x,y_labelPlaceholder:label}
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # 每次不写就会报错


    path = join('ckpt',categoryId)
    if exists(path):
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)

    def singleTrain(string,flag):
        loss,acc,_ = sess.run([cross_entropy,accuracy,train_op],feed_dict=feed_fn(string,flag))
        print('----[acc]----')
        print(acc)
        print('loss:',loss)
        print('-------------')

    for i in range(2000):
        entry = choice(entryListYes)
        string = entry[1]
        singleTrain(string,False)

        entry = choice(entryListNo)
        string = entry[1]
        singleTrain(string,False)

    saver.save(sess, 'ckpt/model.ckpt')


# train()
# prediction()

