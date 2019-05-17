import numpy as np
import tensorflow as tf

import timeit

def separable_conv2d_naive(C, H, W, R_H, R_W, F, input_t, dweights, pweights):
    out_t = np.empty([F, H, W]) 
    dout_t = np.zeros([C, H, W])
    padded_t = np.pad(input_t[0],
                     ((0,0), (int((R_H-1)/2), int((R_H-1)/2+0.5)),
                             (int((R_W-1)/2), int((R_W-1)/2+0.5))),
                     mode='constant')
    for c in range(C):
        for h in range(H):
            for w in range(W):
                dout_t[c, h, w] = np.sum(
                       padded_t[c, h:(h+R_H), w:(w+R_W)] * dweights[c, :, :])
    for f in range(F):
        for h in range(H):
            for w in range(W):
                out_t[f, h, w] = np.dot(dout_t[:, h, w], pweights[f]) 
    return out_t

def init_separable_conv2d_tf(C, H, W, R_H, R_W, F):
    input_ = tf.placeholder(shape=[1,C,H,W], dtype=tf.float32)
    dfilter = tf.placeholder(shape=[R_H, R_W, C, 1], dtype=tf.float32)
    pfilter = tf.placeholder(shape=[1, 1, C, F], dtype=tf.float32)
    strides = [1, 1, 1, 1]
    out = tf.nn.separable_conv2d(input_, dfilter, pfilter,
                                 strides, padding='SAME', name='tf_sepconv2d',
                                 data_format='NCHW')
    sess = tf.Session()
    def separable_conv2d_tf(input_t, dweights, pweights):
        feed_dict = {
                input_: input_t,
                dfilter: np.moveaxis(dweights, 0, -1).reshape((R_H, R_W, C, 1)),
                pfilter: np.transpose(pweights).reshape((1, 1, C, F))
        }
        return sess.run(out, feed_dict=feed_dict)
    return separable_conv2d_tf

if __name__ == '__main__':
    input_t= np.array([[
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ]])  
    dweights = np.array([
            [[1, 1], [1, 1], [1, 1]],
            [[2, 5], [0, 10], [1, 1]],
            [[3, 4], [-1, -3],[3, 3]],
            ], dtype=np.float32)
    pweights = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

    _, C, H, W = input_t.shape
    _, R_H, R_W =  dweights.shape
    F, _ = pweights.shape

    print('C', '\t', 'H', '\t', 'W', '\t', 'R_H', '\t', 'R_W', '\t', 'F')
    print(C, '\t', H, '\t', W, '\t', R_H, '\t', R_W, '\t', F)


    separable_conv2d_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)
    out_tf = separable_conv2d_tf(input_t, dweights, pweights)
    out_naive = separable_conv2d_naive(C, H, W, R_H, R_W, F, input_t, dweights, pweights)
    assert (out_tf == out_naive).all()
    setup = '''
import numpy as np
from __main__ import init_separable_conv2d_tf
from __main__ import separable_conv2d_naive
input_t= np.array([[
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]])  
dweights = np.array([
        [[1, 1], [1, 1], [1, 1]],
        [[2, 5], [0, 10], [1, 1]],
        [[3, 4], [-1, -3],[3, 3]],
        ], dtype=np.float32)
pweights = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

_, C, H, W = input_t.shape
_, R_H, R_W =  dweights.shape
F, _ = pweights.shape

print('C', '\t', 'H', '\t', 'W', '\t', 'R_H', '\t', 'R_W', '\t', 'F')
print(C, '\t', H, '\t', W, '\t', R_H, '\t', R_W, '\t', F)


separable_conv2d_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)
    '''
    time_tf = timeit.timeit('separable_conv2d_tf(input_t, dweights, pweights)',
           setup=setup, number=10000) 
    time_naive = timeit.timeit(
        'separable_conv2d_naive(C, H, W, R_H, R_W, F, input_t, dweights, pweights)',
        setup=setup, number=10000)
    print("time_tf:", time_tf)
    print("time_naive:", time_naive)


