from typing import NamedTuple

import chex
from jax import lax
from jax import numpy as jnp
from jax import random as jr


# pylint: disable=missing-class-docstring,too-few-public-methods
class _DataLoader:
    def __init__(self, num_batches, idxs, get_batch):
        self.num_batches = num_batches
        self.idxs = idxs
        self.get_batch = get_batch

    def __call__(self, idx, idxs=None):
        if idxs is None:
            idxs = self.idxs
        return self.get_batch(idx, idxs)


# pylint: disable=too-many-locals
def as_batch_iterators(
    seed: chex.PRNGKey,
    data: NamedTuple,
    batch_size: int = 64,
    train_val_split: float = 0.9,
    shuffle: bool = True,
):
    """
    Create iterators of subsets of a data set

    Parameters
    ----------
    seed: chex.PRNGKey
        a random key
    data: NamedTuple
        a data set that consists of multiple named elements
    batch_size: int
        size of each batch an iterator returns
    train_val_split: float
        a fraction determining the relative size of the training iterator
    shuffle: bool
        boolean if data should be shuffled along first axis

    Returns
    -------
    Tuple[Iterator, Iterator]
        returns two iterators
    """

    n = data[0].shape[0]
    ctor = data.__class__

    idx_key, seed = jr.split(seed)
    idxs = jr.permutation(idx_key, jnp.arange(n))
    if shuffle:
        data = ctor(*[el[idxs] for _, el in enumerate(data)])

    n_train = int(n * train_val_split)
    y_train = ctor(*[el[:n_train, :] for el in data])
    y_val = ctor(*[el[n_train:, :] for el in data])

    train_rng_key, val_rng_key, seed = jr.split(seed, 3)
    train_itr = as_batch_iterator(train_rng_key, y_train, batch_size, shuffle)
    val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

    return train_itr, val_itr


# pylint: disable=missing-function-docstring
def as_batch_iterator(
    seed: chex.PRNGKey, data: NamedTuple, batch_size: int, shuffle: float
):
    """
    Create a batch iterator

    Parameters
    ----------
    seed: chex.PRNGKey
       a random key
    data: NamedTuple
       a data set that consists of multiple named elements
    batch_size: int
       size of each batch an iterator returns
    shuffle: bool
       boolean if data should be shuffled along first axis

    Returns
    -------
    Iterator
       returns an iterator
    """

    n = data[0].shape[0]
    if n < batch_size:
        num_batches = 1
        batch_size = n
    elif n % batch_size == 0:
        num_batches = int(n // batch_size)
    else:
        num_batches = int(n // batch_size) + 1

    idxs = jnp.arange(n)
    if shuffle:
        idxs = jr.permutation(seed, idxs)

    def get_batch(idx, idxs=idxs):
        start_idx = idx * batch_size
        step_size = jnp.minimum(n - start_idx, batch_size)
        ret_idx = lax.dynamic_slice_in_dim(idxs, idx * batch_size, step_size)
        batch = {
            name: lax.index_take(array, (ret_idx,), axes=(0,))
            for name, array in zip(data._fields, data)
        }
        return batch

    return _DataLoader(num_batches, idxs, get_batch)
