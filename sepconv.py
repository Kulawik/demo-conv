import numpy as np
import tensorflow as tf

def separable_conv2d(C, H, W, R_H, R_W, F, input_t, dweights, pweights):
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
    out_t = np.tensordot(pweights, dout_t, axes=([1],[0]))
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
        return sess.run(out, feed_dict=feed_dict)[0]
    return separable_conv2d_tf
