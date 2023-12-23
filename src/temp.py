#! /usr/bin/env python3
# vim:fenc=utf-8

import argtoml
import equinox as eqx
import jax
import jax.tree_util as jtu
from jo3util.eqx import insert_after, sow

from cnn import CNN
from sae import SAE


O = argtoml.parse_args()

key = jax.random.PRNGKey(O.seed)
key, subkey = jax.random.split(key)

cnn = eqx.tree_deserialise_leaves(O.cnn_storage, CNN(subkey))

sae_hyparam = O.sae[0]
sae_pos = lambda m: m.layers[sae_hyparam.layer]

sae = SAE(sae_hyparam.input_size, sae_hyparam.hidden_size, key)

model = sow(sae_pos, cnn)
model = insert_after(sae_pos, model, sae)
model = sow(sae_pos, model)

freeze_spec = jtu.tree_map(lambda _: False, model)
freeze_spec = eqx.tree_at(
    lambda m: m.layers[sae_hyparam.layer].children[1],
    freeze_spec,
    replace=jtu.tree_map(lambda leaf: eqx.is_array(leaf), sae)
)
# print(model)
print(freeze_spec)
