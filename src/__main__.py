#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path

import argtoml
import equinox as eqx
import jax
import jax.numpy as jnp
import jo3mnist
from jo3util import eqx as jo3eqx
from jo3util.debug import breakpoint as jo3bp
from jo3util.root import run_dir
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .cnn import evaluate
from .data import imagenet
from .googlenet import googlenet
from .sae import SAE, sample_features, train_sae

# Hyperparameters
O = argtoml.parse_args()

key = jax.random.PRNGKey(O.seed)
key, subkey = jax.random.split(key)

if O.debug:
    import debugpy

    debugpy.listen(5678)
    debugpy.wait_for_client()

cnn, cnn_state = googlenet("res/googlenet.pth")

print(
    "accuracy =",
    evaluate(
        eqx.Partial(eqx.nn.inference_mode(cnn), state=cnn_state),
        imagenet(
            "res/imagenet/validation", shuffle=True, batch_size=O.batch_size
        ),
    ),
)

sown_cnn = jo3eqx.sow(lambda m: m.inception5a, cnn)
sown_cnn = eqx.Partial(eqx.nn.inference_mode(sown_cnn), state=cnn_state)

loader = imagenet("res/imagenet/" + O.split, shuffle=False, batch_size=1)
sae_dir = run_dir(O, "run")
directory = sae_dir / O.split
directory.mkdir(parents=True, exist_ok=True)

print("saving features to", directory)
key, subkey = jax.random.split(key)
for i, features in tqdm(
    sample_features(sown_cnn, loader, subkey, directory),
    total=len(loader),
):
    if features is not None:
        jnp.save(directory / f"{i}.npy", features, allow_pickle=False)
