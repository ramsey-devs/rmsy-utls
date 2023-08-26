# pylint: skip-file
from collections import namedtuple

import chex
import pytest
from jax import numpy as jnp
from jax import random as jr

from rmsyutls import as_batch_iterator, as_batch_iterators


@pytest.fixture
def do_shuffle(request):
    return request.param


@pytest.mark.parametrize("do_shuffle", [True, False])
def test_batch_iterator(do_shuffle):
    y, theta = jr.normal(jr.PRNGKey(0), (10, 2)), jnp.zeros((10, 1))
    D = namedtuple("named_dataset", "y theta")(y, theta)
    train_itr = as_batch_iterator(jr.PRNGKey(0), D, 2, do_shuffle)

    chex.assert_equal(train_itr.num_batches, 5)
    chex.assert_equal(len(train_itr(0).keys()), 2)

    for i in range(train_itr.num_batches):
        batch = train_itr(i)["y"]
        subset = y[2 * i : (2 * i + 2)]
        if not do_shuffle:
            chex.assert_trees_all_equal(batch, subset)
        else:
            with pytest.raises(AssertionError):
                chex.assert_trees_all_equal(batch, subset)


def test_batch_iterators():
    named_dataset = namedtuple("named_dataset", "y theta")
    y, theta = jr.normal(jr.PRNGKey(0), (10, 2)), jnp.zeros((10, 1))
    D = named_dataset(y, theta)
    train_itr, val_itr = as_batch_iterators(jr.PRNGKey(0), D, 2, 0.50, False)

    chex.assert_equal(train_itr.num_batches, 3)
    chex.assert_equal(val_itr.num_batches, 3)
    chex.assert_equal(len(train_itr(0).keys()), 2)
    chex.assert_equal(len(val_itr(0).keys()), 2)
