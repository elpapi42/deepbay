import tensorflow as tf
import numpy as np
import pytest

from deepbay import DenseBlock

# This test is not reliable
def test_dense_block_weights_update():
    # seeds the NRG
    tf.random.set_seed(156)

    # testing I/O
    test_input = np.array([[1.0, 1.0, 0.5], [0.1, -1.0, -0.5]])
    test_output = np.array([[0.94], [0.15]])

    # Intantiate a dense block
    block = DenseBlock(units=1, use_batch_norm=True)
    block.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.MeanSquaredError(),
    )
    # Init parameters
    block.train_on_batch(test_input, test_output)

    # Recor initial parameters
    initial_params = block.dense.get_weights()

    # Trin by one step
    block.train_on_batch(test_input, test_output)

    # Record final parameters
    final_params = block.dense.get_weights()

    #Check if the parameters was actually updated
    was_updated = (initial_params[0] != final_params[0]).all() and (initial_params[1] != final_params[1]).all()

    assert was_updated