#! /usr/bin/env python3
# vim:fenc=utf-8

"""

"""

import argtoml
import equinox as eqx
import jax
import jax.numpy as jnp
import mnist
import optax  # https://github.com/deepmind/optax
from jaxtyping import Float  # https://github.com/google/jaxtyping
from jaxtyping import Array, Int, PyTree
from torch.utils.data import DataLoader

from .cnn import CNN, train_cnn
from .sae import SAE, train_sae

# Hyperparameters
O = argtoml.parse_args()
key = jax.random.PRNGKey(O.seed)
key, subkey = jax.random.split(key)

if O.cnn_storage.exists():
    model = eqx.tree_deserialise_leaves(O.cnn_storage, CNN(subkey))
else:
    model = train_cnn(
        subkey,
        O.batch_size,
        O.learning_rate,
        O.steps,
        O.print_every,
        O.cnn_storage,
    )

sae = train_sae(
    key,
    model,
    lambda m: m.layers[6],
    64,
    O.batch_size,
    O.learning_rate,
    O.steps,
    O.print_every,
    O.sae_storage,
)
