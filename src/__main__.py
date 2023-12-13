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
from tqdm import tqdm

from .cnn import CNN, train_cnn
from .sae import SAE, sample_features, train_sae

# Hyperparameters
O = argtoml.parse_args()

key = jax.random.PRNGKey(O.seed)
key, subkey = jax.random.split(key)

if (Path(".") / O.cnn_storage).exists():
    cnn = eqx.tree_deserialise_leaves(O.cnn_storage, CNN(subkey))
else:
    cnn = train_cnn(
        subkey,
        O.batch_size,
        O.learning_rate,
        O.steps,
        O.print_every,
        O.cnn_storage,
    )
    eqx.tree_serialise_leaves(O.cnn_storage, cnn)

for sae_hyperparams in O.sae:
    sae_dir = run_dir(sae_hyperparams)
    if sae_dir.exists():
        continue

    sae = train_sae(
        key,
        cnn,
        lambda m: m.layers[sae_hyperparams.layer],
        sae_hyperparams.input_size,
        sae_hyperparams.hidden_size,
        O.batch_size,
        sae_hyperparams.learning_rate,
        O.steps,
        O.print_every,
    )

    sae_dir.mkdir()
    argtoml.save(sae_hyperparams, sae_dir / "sae-hyperparams.toml")
    argtoml.save(O, sae_dir / "config.toml")
    jo3eqx.save(
        sae_dir / f"sae.eqx",
        sae,
        {
            "in_size": sae_hyperparams.input_size,
            "hidden_size": sae_hyperparams.hidden_size,
        },
    )

for sae_hyperparams in O.sae:
    sae_dir = run_dir(sae_hyperparams)
    sae = jo3eqx.load(sae_dir / f"sae.eqx", SAE)
    sown_cnn = jo3eqx.sow(lambda m: m.layers[sae_hyperparams.layer], cnn)
    trainloader, testloader = jo3mnist.load(
        batch_size=O.batch_size, shuffle=False
    )

    train_dir = sae_dir / f"train"
    if not train_dir.exists():
        print("saving features from the training set")
        train_dir.mkdir()
        for i, features in tqdm(
            sample_features(sown_cnn, sae, trainloader), total=len(trainloader)
        ):
            jnp.save(train_dir / f"{i}.npy", features, allow_pickle=False)

    test_dir = sae_dir / f"test"
    if not test_dir.exists():
        print("saving features from the test set")
        test_dir.mkdir()
        for i, features in tqdm(
            sample_features(sown_cnn, sae, testloader), total=len(testloader)
        ):
            jnp.save(test_dir / f"{i}.npy", features, allow_pickle=False)
