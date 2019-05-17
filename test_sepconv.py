import numpy as np
from sepconv import init_separable_conv2d_tf
from sepconv import separable_conv2d

import pytest

def test_separable_conv2d_example():
    input_t= np.array([[
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        ]])  
    dweights = np.array([
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            ], dtype=np.float32)
    pweights = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]])
    _, C, H, W = input_t.shape
    _, R_H, R_W =  dweights.shape
    F, _ = pweights.shape
    out = separable_conv2d(C, H, W, R_H, R_W, F, input_t, dweights, pweights)
    assert out.shape == (4, 3, 3) 
    assert (out[0] == [
            [4, 6, 4],
            [6, 9, 6],
            [4, 6, 4]
            ]).all()
    assert (out[1] == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
            ]).all()
    assert (out[2] == [
            [-8, -12, -8],
            [-12, -18, -12],
            [-8, -12, -8]
            ]).all()
    assert (out[3] == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
            ]).all()

def test_separable_conv2d_tf_example():
    input_t= np.array([[
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        ]])  
    dweights = np.array([
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            ], dtype=np.float32)
    pweights = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]])
    _, C, H, W = input_t.shape
    _, R_H, R_W =  dweights.shape
    F, _ = pweights.shape
    conv2d_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)
    out = conv2d_tf(input_t, dweights, pweights)
    assert out.shape == (4, 3, 3) 
    assert (out[0] == [
            [4, 6, 4],
            [6, 9, 6],
            [4, 6, 4]
            ]).all()
    assert (out[1] == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
            ]).all()
    assert (out[2] == [
            [-8, -12, -8],
            [-12, -18, -12],
            [-8, -12, -8]
            ]).all()
    assert (out[3] == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
            ]).all()

@pytest.mark.parametrize('input_t, dweights, pweights, expected',[
        ( # set 0, in 3x3x3, f 3x3, out 4
        np.array([[  #  input_t 
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]], 
        ]], dtype=np.float32),
        np.array([#  dweights
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        ], dtype=np.float32),
        np.array([#  pweights
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]
        ], dtype=np.float32),
        np.array([#  expected
            [[4, 6, 4],
             [6, 9, 6],
             [4, 6, 4]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[-8, -12, -8],
             [-12, -18, -12],
             [-8, -12, -8]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ], dtype=np.float32),
        ),

        ( # set 1, in 3x3x3, f 2x2, out 3
        np.array([[  #  input_t 
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]], 
        ]], dtype=np.float32),
        np.array([#  dweights
        [[1, 2], [4, 5]],
        [[1, 1], [1, 1]],
        [[-1, -1], [-1, -1]],
        ], dtype=np.float32),
        np.array([#  pweights
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float32),
        np.array([#  expected
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[4, 4, 2],
             [4, 4, 2],
             [2, 2, 1]],
            [[-8, -8, -4],
             [-8, -8, -4],
             [-4, -4, -2]]
        ], dtype=np.float32),
        ),
        
        ( # set 2, in 3x3x3, f 3x2, out 3
        np.array([[  #  input_t 
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]], 
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]], dtype=np.float32),
        np.array([#  dweights
        [[1, 1], [1, 1], [1, 1]],
        [[-1, -1], [-1, -1], [-1, -1]],
        [[1, 2], [4, 5], [6, 7]],
        ], dtype=np.float32),
        np.array([#  pweights
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float32),
        np.array([#  expected
            [[4, 4, 2],
             [6, 6, 3],
             [4, 4, 2]],
            [[-8, -8, -4],
             [-12, -12, -6],
             [-8, -8, -4]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
        ], dtype=np.float32),
        ),
        
        ( # set 3, in 1x3x3, f 3x3, out 3
        np.array([[  #  input_t 
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ]], dtype=np.float32),
        np.array([#  dweights
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        ], dtype=np.float32),
        np.array([#  pweights
        [1], [0], [2]
        ], dtype=np.float32),
        np.array([#  expected
            [[-4, -6, -4],
             [-6, -9, -6],
             [-4, -6, -4]],
            [[-0, -0, -0],
             [-0, -0, -0],
             [-0, -0, -0]],
            [[-8, -12, -8],
             [-12, -18, -12],
             [-8, -12, -8]],
        ], dtype=np.float32),
        ),
        
        ( # set 4, in 1x2x2, f 3x3, out 3
        np.array([[  #  input_t 
        [[1, 2], [3, 4]],
        ]], dtype=np.float32),
        np.array([#  dweights
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ], dtype=np.float32),
        np.array([#  pweights
        [1], [0], [2]
        ], dtype=np.float32),
        np.array([#  expected
            [[27, 27],
             [17, 17]],
            [[0, 0],
             [0, 0]],
            [[54, 54],
             [34, 34]],
        ], dtype=np.float32),
        ),
])
def test_separable_conv2d_expected_and_against_tf(input_t, dweights, pweights, expected):
    _, C, H, W = input_t.shape
    _, R_H, R_W =  dweights.shape
    F, _ = pweights.shape
    out = separable_conv2d(C, H, W, R_H, R_W, F, input_t, dweights, pweights)
    out_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)(input_t, dweights, pweights)
    assert (out == out_tf).all()
    assert (out == expected).all()

@pytest.mark.parametrize('seed, input_t_shape, dweights_shape, pweights_shape', [
    (0, (1, 3, 16, 16), (3, 1, 1), (10, 3)),
    (1, (1, 3, 3, 3), (3, 3, 3), (10, 3)),
    (2, (1, 5, 256, 256), (5, 5, 5), (10, 5)),
    (3, (1, 5, 256, 256), (5, 3, 3), (10, 5)),
    (4, (1, 5, 256, 256), (5, 2, 2), (10, 5)),
    (5, (1, 1, 256, 128), (1, 3, 3), (10, 1)),
    (6, (1, 1, 128, 256), (1, 3, 3), (10, 1)),
    (7, (1, 1, 7, 13), (1, 5, 7), (10, 1)),
    (8, (1, 1, 7, 13), (1, 5, 7), (10, 1)),
    (9, (1, 1, 1, 1), (1, 3, 3), (10, 1)),
])
def test_separable_conv2d_against_tf_random(
        seed, input_t_shape, dweights_shape, pweights_shape):
    np.random.seed(seed)
    input_t = np.random.rand(*input_t_shape)
    dweights = np.random.rand(*dweights_shape)
    pweights = np.random.rand(*pweights_shape)
    _, C, H, W = input_t.shape
    _, R_H, R_W =  dweights.shape
    F, _ = pweights.shape
    out = separable_conv2d(C, H, W, R_H, R_W, F, input_t, dweights, pweights)
    out_tf = init_separable_conv2d_tf(C, H, W, R_H, R_W, F)(input_t, dweights, pweights)
    np.testing.assert_allclose(out, out_tf, rtol=1e-6)

