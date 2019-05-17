import numpy as np
import tensorflow as tf
import timeit
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sepconv import init_separable_conv2d_tf
from sepconv import separable_conv2d

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

    separable_conv2d_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)
    out_tf = separable_conv2d_tf(input_t, dweights, pweights)
    out= separable_conv2d(C, H, W, R_H, R_W, F, input_t, dweights, pweights)
    assert (out_tf == out).all()

    setup = '''
import numpy as np
from sepconv import init_separable_conv2d_tf
from sepconv import separable_conv2d
input_t = np.random.rand(1, 3, 128, 128)
dweights = np.random.rand(3, 3, 3)
pweights = np.random.rand(5, 3)
_, C, H, W = input_t.shape
_, R_H, R_W =  dweights.shape
F, _ = pweights.shape
separable_conv2d_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)
    '''
    time_tf = timeit.timeit('separable_conv2d_tf(input_t, dweights, pweights)',
                            setup=setup, number=7) 
    time_custom = timeit.timeit(
        'separable_conv2d(C, H, W, R_H, R_W, F, input_t, dweights, pweights)',
        setup=setup, number=7)
    print("time_tf:", time_tf)
    print("time_custom", time_custom)


