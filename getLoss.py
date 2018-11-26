import tensorflow as tf
import numpy as np

CE = tf.nn.sparse_softmax_cross_entropy_with_logits
def getLoss(y):
    y_label = tf.placeholder(tf.int32, shape=[None], name='y_label')

    y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)
    print('Output Y.shape:', y_predict.shape)    

    # tf.summary.histogram('y_predict', y_predict)

    # y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    # print('Y Label Reshape:', y_label_reshape)

    correct_prediction = tf.equal(y_predict, y_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', accuracy)

    print('Prediction:', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
    cross_entropy = tf.reduce_mean(CE(labels=y_label,logits=tf.cast(y, tf.float32)))
    
    # tf.summary.scalar('loss', cross_entropy)
    print('cross_entropy:',cross_entropy)

    return y_label,cross_entropy,accuracy
