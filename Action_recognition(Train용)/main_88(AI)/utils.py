import tensorflow as tf
import os
import numpy as np
import math
import csv
import time

PI = 3.1415926
def vector_size(x):
    if x.ndim == 2 :
        return np.sqrt(np.sum(x*x,axis=1))
    if x.ndim == 1:
        return np.sqrt(np.sum(x*x))
    else:
        print('내가 생각한 차원에서 벡터 사이즈 계산한거 달라용')
        return 0

def outer_product(x,y):
    V1 = np.array([x[1]*y[2], x[0]*y[2], x[0]*y[1]])
    V2 = np.array([y[1]*x[2], y[0]*x[2], y[0]*x[1]])
    return V1-V2

def v2v_angle(x,y):
    if(vector_size(x) == 0 or vector_size(y) ==0): #0으로 나눠지는것을 방지하기 위해.
        sin = vector_size(outer_product(x, y)) / (vector_size(x) * vector_size(y) + 0.0000000000001)
    else:
        sin = vector_size(outer_product(x,y)) / (vector_size(x) * vector_size(y))
    return (math.asin(sin)) * (180/PI)

def ckpt_load(sess,checkpoint_dir, saver):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('[*] Loaded weight...', ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False



def load_data(subj):
    t1 = time.time()

    data_fname = './data/pose_data for labeling.csv'
    pose_data = np.loadtxt(data_fname, skiprows=1, delimiter=',', usecols=list(range(3, 60)))
    label_data = np.loadtxt(data_fname, skiprows=1, delimiter=',', usecols=list(range(61, 62)))

    print("Data load Time : [%4.4f]" % (time.time()-t1))

    if (subj == 1):
        test_x = pose_data[:24637]
        target = label_data[:24637].astype(int)

    elif (subj == 2):
        test_x = pose_data[24637:]
        target = label_data[24637:].astype(int)

    L_thumb = test_x[:, 18:21]
    R_thumb = test_x[:, 30:33]
    data = np.append(L_thumb, R_thumb, axis=1).reshape(-1,2,3)
    n_frame = data.shape[0]
    t_interval = 1 # sec
    input = np.array([])

    t2 = time.time()
    for i in range(n_frame):
        pre_frame = np.array([[.0, .0, .0], [.0, .0, .0]])
        cur_frame = data[i]
        velocity = (pre_frame - cur_frame) / t_interval #shape = (2,3) # 스무스하게 하기

        L_v, R_v = vector_size(velocity)
        L_dir, R_dir = velocity/np.array(vector_size(velocity)).reshape(2,1)
        angle =  v2v_angle(cur_frame[0], cur_frame[1]) / 180

        element = np.array([L_v, R_v , angle])
        element = np.append(element, L_dir)
        element = np.append(element, R_dir)  # 왼오속력, 각도 왼오x,y,z #(9,)
        input = np.append(input, element)

        pre_frame = cur_frame

    input = input.reshape(-1,9)  # (frame, element)
    print("데이터 각도 뽑는거 반복문 걸린시간 : [%4.4f]" % (time.time()-t2))
    print('[*] Data Load ' )
    return input, test_x, target  # input:model2, test_x:model1

def cal_accuracy(m1_x, m2_x, target, sess, accuracy, correct_prediction, hypothesis, m1_X, m2_X, Y):
    n_data = m1_x.shape[0]
    _acc = np.array([])
    n_class = 7
    batch_size = 900
    error_data = np.array([])

    n_batch = n_data // batch_size
    for j in range(n_batch):  # 배치만큼 빼는 구간.
        x1 = np.array([])
        x2 = np.array([])
        y = np.array([])
        if (j * batch_size + batch_size > n_data):
            break
        m1_batch_x = m1_x[j * batch_size:j * batch_size + batch_size] # batch_size 만큼 빼낸다.
        m2_batch_x = m2_x[j * batch_size:j * batch_size + batch_size] # batch_size 만큼 빼낸다.
        batch_y = target[j * batch_size:j * batch_size + batch_size]

        count = 0
        for i in range(batch_size):
            left_idx = i - 15
            right_idx = i + 15
            if left_idx >= 0 and right_idx < batch_size:
                count += 1
                x1 = np.append(x1, m1_batch_x[left_idx:right_idx])
                x2 = np.append(x2, m2_batch_x[left_idx:right_idx])
                y = np.append(y, np.eye(n_class)[int(batch_y[right_idx - 1])])

                if (count == 100): # LSTM batch_size = 100
                    x1 = x1.reshape(count, 30, 57)
                    x2 = x2.reshape(count, 30, 9)
                    y = y.reshape(count, n_class)
                    #acc, _correct_prediction, _hypothesis = sess.run([accuracy,correct_prediction, hypothesis], feed_dict={m1_X: x1, m2_X: x2, Y: y}) #주석
                    acc = sess.run(accuracy, feed_dict={m1_X: x1, m2_X: x2, Y: y})

                    _acc= np.append(_acc, acc)

                    '''
                     ##
                    for k in range(_correct_prediction.shape[0]):
                        if _correct_prediction[k] == False:
                            error_data = np.append(error_data, x1[k])
                            error_data = np.append(error_data, np.argmax(_hypothesis[k]))
                            error_data = np.append(error_data, np.argmax(y[k]))  # error_data = [좌표, 예측, 실제]
                            print(error_data)
                    ##


                    x1 = np.array([])
                    x2 = np.array([])
                    y = np.array([])
                    count = 0
                    '''

    '''
    ##

    error_data = error_data.reshape(-1, 30 * 57 + 2)
    print(error_data)
    f_name = "error.csv"
    f = open(f_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range(error_data.shape[0]):
        wr.writerow(error_data[i])
    f.close()

    ##
    '''




    return np.sum(_acc)/_acc.shape[0]