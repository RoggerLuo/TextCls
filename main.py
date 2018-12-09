import tensorflow as tf
import numpy as np
from model import getTrainingModel,getPredictionModel
from getLoss import getLoss
from embed import str2embed
from os.path import exists
from tensorflow.contrib import layers
from random import choice
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

    string = '先搞课程模块,登陆界面和状态栏以后再搞,杨老师那边先丢点东西过去'
    _y = sess.run(y,feed_dict=predict_feed_fn(string))
    print(_y)
    string = '每天的节点要搞清楚,自己的,公司的,特别是自己的,都是焦虑的来源'
    _y = sess.run(y,feed_dict=predict_feed_fn(string))
    print(_y)

    string = '不要把决策权给客户，解决问题的方向是:对用户有帮助的,对方说的不一定能解决问题自己判断，别照单全收，而是给出建议'
    _y = sess.run(y,feed_dict=predict_feed_fn(string))
    print(_y)


    string = '当外界不能按照主观意愿发展的时候，人就会产生一系列特殊的行为'
    _y = sess.run(y,feed_dict=predict_feed_fn(string))
    print(_y)
    
    string = '前端工程化,应用生命周期各个阶段，组件化强制规范,文件组织结构与框架。'
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

    yes=[
        '多搜索一下资讯',
        '别急着动手,要动手 也别太狭隘，动手练习一下而已',
        "总结当天,写readme.md",
        'Demon这类临时项目,代码改了哪里,都要记下来,以免未来忘记',
        '工作时间不要聊微信,这里的工作时间是广泛的,包括我的，和工作的,无论是哪个，效率都很重要',
        '翻原来的效率',
        '噪音太多,把这一大堆事放在自己肩上,我想做的到底是什么呢,专注呢',

        '先找“确定的”东西,从 好界定的 到模糊的,先确定好 “一定确定”的东西',
        '自己要做的事,圆圆的事不要耽误了自己的事，这就不好了',

        '提前预测，再来复盘',
            
        '评估时间，不行就及时切换方案，',
        '肯定是有代替方案的，总结起来',


        '全局掌控',

        '答应、确定下来的事情记录在日历上，2小时看一次日历，回顾计划',

        '预留时间，不要安排得太紧，',

        '计划清楚：弄清楚deadline，一开始花时间把计划的各个部分、阶段过一遍，',

        '隐形成本：沟通、协调和等待响应的成本,随着事物的发展，一定比率的细节 肯定考虑不到 的成本',

        '协作,及时、分阶段反馈情况',
        '记忆检索是一种快思考的过程，很省力，',
        '符合认知的最小行动、最经济的原则，',
        '但能解决大部分问题。'

    ]
    no = [
        '三个仓和一个仓没有区别',
        '短期就是无法预测',
        '不要挑战无法预测的事情',
        '中年男子说：谢谢！这次地球之行确实给我留下了美好的印象，这是我们的祖庭呀，我会把这些印象永远保留在心中。',
        '小伙子热情地说：“地球太美了，地球人太热情了！我真想在这儿再多待几年……”',
        '杰弗里插了一句：“你已经在地球逗留了三年，你们四位都是。”',
        '老年男人说：“对，我们真舍不得走。特别是我，这恐怕是我最后一次返回故土了。”他的声调中透着苍凉。',
        '姑娘兴高采烈地说：“地球人非常可爱，尤其是男人们，可惜我没能带走一个如意郎君。”',
        '杰弗里笑道：“不过，据我所知，已经有几位地球小伙子追求过你了，对吧！”',
        'Redux与react component错位问题',
        '包含state组件的item，',
        '改变redux数据之后，比如，在redux动态增加数组元素(item)之后，items数量变化，',
        '可能出现错位：',
        '答：硬实时任务是指系统必须满足任务对截止时间的要求，否则可能出现难以预测的结果。 ',
        '举例来说，运载火箭的控制等。 ',
        '软实时任务是指它的截止时间并不严格，偶尔错过了任务的截止时间，对系统产生的影 ',
        '响不大。举例：网页内容的更新、火车售票系统。 ',
        '10．在8位微机和16位微机中，占据了统治地位的是什么操作系统？ '
    ]

    # return
    for i in range(200):
        

        string = choice(no) #'这个是圆的分类'
        print(string)
        flag = False
        print(flag)
    # print(sess.run(tes))

        loss,acc,_ = sess.run([cross_entropy,accuracy,train_op],feed_dict=feed_fn(string,flag))
        

        print('----[acc]----')
        print(acc)
        print('loss:',loss)
        print('----------------------------------------')

        # print('这个是圆圆的分类')
        string = choice(yes) #'这个是圆圆的分类'
        print(string)

        flag = True
        print(flag)

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

