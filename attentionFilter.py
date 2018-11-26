import tensorflow as tf
import numpy as np

# num_units = 256
num_units = 200

attention_len = 512

def matU(shape=[num_units,attention_len], stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        return tf.get_variable('matU',initializer=initial)


def queryW(shape=[attention_len], value=0.1):
    initial = tf.constant(value=value, shape=shape)
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        return tf.get_variable('queryW',initializer=initial)

    # return tf.Variable(initial)

def queryV(shape=[attention_len,1], value=0.1):
    initial = tf.constant(value=value, shape=shape)
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        return tf.get_variable('queryV',initializer=initial)

    # return tf.Variable(initial)




def attentionModel(embedingPlaceholder):

    # input_embed = tf.reshape(embedingPlaceholder,[1,-1,200])
    # with tf.variable_scope('encode'):

    #     cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    #     cell2 = tf.contrib.rnn.GRUCell(num_units=num_units)
    #     encoder = tf.contrib.rnn.MultiRNNCell([cell,cell2])
    #     encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder, input_embed, dtype=tf.float32)
    #     # shape=(1, ?, 256)
    #     encoder_outputs = tf.reshape(encoder_outputs,[-1,256])

    with tf.variable_scope('attention'):
        U = matU()
        W = queryW()
        V = queryV()
        activation = tf.tanh(tf.matmul(embedingPlaceholder, U) + W)
        # activation = tf.tanh(tf.matmul(encoder_outputs, U) + W) 
        value = tf.matmul(activation,V) # shape=(?, 1)
        flatValue = tf.reshape(value,[-1]) # shape=(?)
        scores = tf.nn.softmax(flatValue)
    return scores

# def demoRun():
#     scores = attentionModel()
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer()) # 每次不写就会报错
#     value = sess.run(scores,feed_dict={'embeding:0':np.random.random((10,200))})
#     print(value)
    """ [0.09600932 0.10093873 0.04859696] """
# demoRun()

def getArticleRepresentation(embedingPlaceholder): # 加权平均
    scores = attentionModel(embedingPlaceholder)
    reshapedScores = tf.reshape(scores,[-1,1]) 
    Cs = embedingPlaceholder*reshapedScores
    C = tf.reduce_sum(Cs, 0)
    # print(C) # shape=(200,)
    return C

# def getW(name,shape=[200,200], stddev=0.1, mean=0):
#     initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
#     with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
#         return tf.get_variable(name,initializer=initial)

#     # return tf.Variable(initial)

# def getB(name,shape=[200], value=0.1):
#     initial = tf.constant(value=value, shape=shape)
#     # return tf.Variable(initial)
#     with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
#         return tf.get_variable(name,initializer=initial)


# def multilayer_perceptron(x, keep_prob): # 在不go down gradient，keep_prob带来了每次sess.run的唯一的不确定性，
#     x = tf.reshape(x,[1,200])
#     layer_1 = tf.add(tf.matmul(x, getW('w1',[200,200])), getB('b1',[200]))
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.nn.dropout(layer_1, keep_prob)
#     out_layer = tf.matmul(layer_1, getW('w2',[200,2])) + getB('b2',[2])
#     # Tensor("add:0", shape=(1, 2), dtype=float32)
#     return out_layer

# def getPredictionModel():
#     rep = getArticleRepresentation()
#     y = multilayer_perceptron(rep,1)
#     soft_y = tf.nn.softmax(y)
#     return [embedingPlaceholder,soft_y]

# def getTrainingModel():
#     rep = getArticleRepresentation()
#     y = multilayer_perceptron(rep,0.8)
#     return [embedingPlaceholder,y]
#     """
#     embeding [?,200]
#     y [1,2]
#     """
