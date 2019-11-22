import tensorflow as tf
import numpy as np
import pytest

from DeepBay import DenseBlock

def test_dense_block_call():
    # Set seeds for reproducibility
    tf.random.set_seed(156)

    # I/O
    test_input = np.array([[1.0, 1.0, 0.5], [0.1, -1.0, -0.5]])
    test_output = np.array([[0.14267592], [0.23824959]])

    # Intantiate a dense block
    dense = DenseBlock(1)
    dense.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.MeanSquaredError(),
    )
    dense.train_on_batch(test_input, np.array([[0.0], [0.0]]))

    assert tf.equal(dense(test_input), test_output).numpy().all()
