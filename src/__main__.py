#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path

import argtoml
import equinox as eqx
import jax
import jax.numpy as jnp
import jo3mnist
from eqxvision.models import googlenet
from jo3util.debug import breakpoint as jo3bp
from jo3util import eqx as jo3eqx
from jo3util.root import run_dir
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .cnn import evaluate
from .sae import SAE, sample_features, train_sae
from .data import imagenet

# Hyperparameters
O = argtoml.parse_args()

key = jax.random.PRNGKey(O.seed)
key, subkey = jax.random.split(key)

# print("accuracy =", evaluate(
#     googlenet("res/googlenet.pth"),
#     imagenet(
#         "res/imagenet/validation", shuffle=True, batch_size=O.batch_size
#     )
# )

sown_cnn = jo3eqx.sow(lambda m: m.inception5a, googlenet("res/googlenet.pth"))

loader = imagenet("res/imagenet/train", shuffle=False, batch_size=1)
sae_dir = run_dir(O, "run")
train_dir = sae_dir / f"train"
train_dir.mkdir(parents=True, exist_ok=True)

print("saving features from the training set")
key, subkey = jax.random.split(key)
for i, features in tqdm(
    sample_features(sown_cnn, loader, subkey, train_dir),
    total=len(loader),
):
    if features is not None:
        jnp.save(train_dir / f"{i}.npy", features, allow_pickle=False)

loader = imagenet("res/imagenet/validation", shuffle=False, batch_size=1)
test_dir = sae_dir / f"test"
test_dir.mkdir(exist_ok=True)

print("saving features from the test set")
key, subkey = jax.random.split(key)
for i, features in tqdm(
    sample_features(sown_cnn, loader, subkey, test_dir),
    total=len(loader),
):
    if features is not None:
        jnp.save(test_dir / f"{i}.npy", features, allow_pickle=False)
