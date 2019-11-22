import tensorflow as tf
import numpy as np
import pytest

from DeepBay import DenseBlock

def test_dense_block_call():
    # Set seeds for reproducibility
    tf.random.set_seed(0)

    # I/O
    test_input = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    test_output = np.array([[0.0], [0.0]])

    # Intantiate a dense block
    dense = DenseBlock(1)

    assert tf.equal(dense(test_input), test_output).numpy().any()
    