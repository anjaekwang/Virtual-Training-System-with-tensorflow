import tensorflow as tf
import os
import numpy as np
import math

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
    if(vector_size(x) == 0 or vector_size(y) ==0):  # 0으로 나눠지는것을 방지
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

def data(test_x):
    L_thumb = test_x[:, 18:21]
    R_thumb = test_x[:, 30:33]
    data = np.append(L_thumb, R_thumb, axis=1).reshape(-1,2,3)
    n_frame = data.shape[0]
    t_interval = 1  #sec
    input = np.array([])

    for i in range(n_frame):
        pre_frame = np.array([[.0, .0, .0], [.0, .0, .0]])
        cur_frame = data[i]
        velocity = (pre_frame - cur_frame) / t_interval

        L_v, R_v = vector_size(velocity)
        L_dir, R_dir = velocity/np.array(vector_size(velocity)).reshape(2,1)
        angle =  v2v_angle(cur_frame[0], cur_frame[1]) / 180

        element = np.array([L_v, R_v , angle])
        element = np.append(element, L_dir)
        element = np.append(element, R_dir)  # 왼오속력, 각도 왼오x,y,z #(9,)
        input = np.append(input, element)

        pre_frame = cur_frame

    return input
