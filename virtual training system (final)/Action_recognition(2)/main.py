import tensorflow as tf
import numpy as np
from utils import ckpt_load, data
from socket import *
import threading
import os

class SocketInfo():
    HOST = "127.0.0.1"
    PORT = 1234
    BUFSIZE = 630
    ADDR = (HOST, PORT)


class socketInfo(SocketInfo):
    HOST = "127.0.0.1"

def skeleton_socket():
    global skeleton_data
    skeleton_data = np.array([])
    while (True):
        csock = socket(AF_INET, SOCK_STREAM)
        csock.connect(socketInfo.ADDR)

        commend = csock.recv(socketInfo.BUFSIZE, MSG_WAITALL)
        data = np.array(commend.decode("UTF-8").partition('\n')[0].split(', '))

        try:
            if(skeleton_data.ndim == 2 and skeleton_data.shape[0] == 30):
                skeleton_data = skeleton_data[1:skeleton_data.shape[0]] # slice는 shape 안 망가짐

            for i in range(len(data)):
                skeleton_data = np.append(skeleton_data, float(data[i]))
            skeleton_data = skeleton_data.reshape(-1,60)


        except ValueError:
            print("Skeleton pose를 얻을수 없습니다. Kinect범위내 들어와 주십시오.")  #  바꿀필요 있다 소켓이안올때 값안들어오는듯
            if(skeleton_data.shape[0] != 0):
                skeleton_data = skeleton_data[1:skeleton_data.shape[0]]

        except ConnectionResetError:
            print("연결이 끊겼습니다. 프로그램을 종료합니다.")
            exit()

        finally:
            csock.close()

learning_rate = 0.001
n_class = 7
m1_hidden_size = 57  # 좌표수 =  19부위 (원점제외 )
m2_hidden_size = 9
batch_size = 1
keep_prob = 1  #train 0.7 test 1

'''
model1의 input : 30개의 frame 단위 별 각 skeleton 20개부위 * 3개 xyz좌표 -3 (원점은 항상 0이므로 빼준다.)
model2의 input : 30개의 frame 단위 별 9개의 특정 요소 (왼쪽손 속력 , 오른쪽손 속력, 왼쪽오른쪽손과 가슴이 이루는 각도, 
                 왼속 xyz 좌표, 오른손 xyz 좌표) 
output : label   
'''
m1_X = tf.placeholder(tf.float32, shape=[None,30,57])
m2_X = tf.placeholder(tf.float32, shape=[None,30,9])
Y = tf.placeholder(tf.float32, shape=[None,n_class])

# model1
with tf.variable_scope('lstm1'):
    m1_cell = tf.contrib.rnn.BasicLSTMCell(num_units=m1_hidden_size, state_is_tuple=True)
    m1_initial_state = m1_cell.zero_state(batch_size, tf.float32)
    m1_outputs, _m1_states = tf.nn.dynamic_rnn(m1_cell, m1_X, initial_state=m1_initial_state, dtype=tf.float32)
    m1_outputs = tf.nn.relu(m1_outputs[:,-1]) # 마지막 결과만 사용 shape = (batch_size,hidden_size =57)
    m1_outputs = tf.nn.dropout(m1_outputs, keep_prob=keep_prob)
    m1_outputs = tf.reshape(m1_outputs, shape=[-1,19,3,1])  # shape = 100,19,3,1

    model1_LSTM_hist = tf.summary.histogram("m1_outputs", m1_outputs)

with tf.name_scope("layer1") as scope:
    w3 = tf.get_variable("w3", shape=[19 * 3 * 1, n_class], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', shape=[1], initializer=tf.contrib.layers.xavier_initializer())
    m1_outputs = tf.layers.flatten(m1_outputs)  # shape = 100, 5
    m1_hypothesis = tf.nn.relu(tf.matmul(m1_outputs, w3) + b)
    m1_hypothesis = tf.nn.dropout(m1_hypothesis, keep_prob=keep_prob)

    w3_hist = tf.summary.histogram("weights3", w3)
    b1_hist = tf.summary.histogram("biases1", b)
    m1_hypothesis_hist = tf.summary.histogram("m1_hypothesis", m1_hypothesis)


# model2
with tf.variable_scope('lstm2'):
    m2_cell = tf.contrib.rnn.BasicLSTMCell(num_units=m2_hidden_size, state_is_tuple=True)
    m2_initial_state = m2_cell.zero_state(batch_size, tf.float32)
    m2_outputs, _m2_states = tf.nn.dynamic_rnn(m2_cell, m2_X, initial_state=m2_initial_state, dtype=tf.float32)
    m2_outputs = tf.nn.relu(m2_outputs[:,-1]) # 마지막 결과만 사용 shape = (batch_size,hidden_size =9)
    m2_outputs = tf.nn.dropout(m2_outputs, keep_prob=keep_prob)

    model2_LSTM_hist = tf.summary.histogram("m2_outputs", m2_outputs)


with tf.name_scope("layer1_2") as scope:
    w4 = tf.get_variable("w4", shape=[9, n_class], initializer=tf.contrib.layers.xavier_initializer())
    m2_hypothesis = tf.matmul(m2_outputs, w4)
    m2_hypothesis = tf.nn.dropout(m2_hypothesis, keep_prob=keep_prob)

    w4_hist = tf.summary.histogram("weights2", w4)
    m2_hypothesis_hist = tf.summary.histogram("m2_hypothesis", m2_hypothesis)

# Fully connected
with tf.name_scope("fc") as scope:
    w5 = tf.get_variable('w5', shape=[n_class, n_class],initializer=tf.contrib.layers.xavier_initializer())
    w6 = tf.get_variable('w6', shape=[n_class, n_class],initializer=tf.contrib.layers.xavier_initializer())
    hypothesis = (tf.matmul(m1_hypothesis, w5) + tf.matmul(m2_hypothesis, w6)) / 2

    w5_hist = tf.summary.histogram("weights5", w5)
    w6_hist = tf.summary.histogram("weights6", w6)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)


with tf.Session() as sess:
    checkpoint_dir = './checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if ckpt_load(sess, checkpoint_dir, saver):
        print(" [*] Load SUCCESS")
    else:

        print(" [!] Load failed...")

    t = threading.Thread(target=skeleton_socket)
    t.start()

    while(True):
        if(skeleton_data.ndim == 2 and skeleton_data.shape[0] == 30):
            m1_data = skeleton_data[:, 3:].reshape(-1, 30, 57)
            m2_data = data(skeleton_data).reshape(-1, 30, 9)
            predict = sess.run([hypothesis],feed_dict={m1_X: m1_data, m2_X: m2_data})
            print(np.argmax(predict))


