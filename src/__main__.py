#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path

import argtoml
import equinox as eqx
import jax
import jax.numpy as jnp
import jo3mnist
from jo3util import eqx as jo3eqx
from jo3util.root import run_dir
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from eqxvision.models import googlenet

from .cnn import CNN, train_cnn
from .sae import SAE, sample_features, train_sae
from .tiny_imagenet import tiny_imagenet

# Hyperparameters
O = argtoml.parse_args()

key = jax.random.PRNGKey(O.seed)
key, subkey = jax.random.split(key)

cnn = googlenet(
    "https://download.pytorch.org/models/googlenet-1378be20.pth"
)

trainloader, testloader = tiny_imagenet(
    "res/tiny-imagenet-200",
    shuffle=False,
    batch_size=1
)

for sae_hyperparams in O.sae:
    sae_dir = run_dir(sae_hyperparams, "run")

    if O.googlenet:
        sae_pos = lambda m: m.inception5b
    else:
        sae_pos = lambda m: m.layers[sae_hyperparams.layer]

    sown_cnn = jo3eqx.sow(sae_pos, cnn)

    train_dir = sae_dir / f"train"
    print("saving features from the training set")
    key, subkey = jax.random.split(key)
    for i, features in tqdm(
        sample_features(sown_cnn, trainloader, subkey, train_dir), total=len(trainloader)
    ):
        jnp.save(train_dir / f"{i}.npy", features, allow_pickle=False)

    test_dir = sae_dir / f"test"
    test_dir.mkdir(exist_ok=True)
    print("saving features from the test set")
    key, subkey = jax.random.split(key)
    for i, features in tqdm(
        sample_features(sown_cnn, testloader, subkey), total=len(testloader)
    ):
        jnp.save(test_dir / f"{i}.npy", features, allow_pickle=False)
