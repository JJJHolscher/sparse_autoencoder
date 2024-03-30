#! /usr/bin/env python3
# vim:fenc=utf-8

"""

"""

from datetime import datetime
from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jo3mnist
import optax
from jaxtyping import Array, Float, Int, PyTree
from jo3util.eqx import insert_after, sow
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .cnn import cross_entropy


def filter_value_and_grad_with_aux(f):
    return eqx.filter_value_and_grad(f, has_aux=True)


class SAE(eqx.Module):
    we: Float
    wd: Float
    be: Float
    bd: Float

    def __init__(self, in_size, hidden_size, key=jax.random.PRNGKey(42)):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        initializer = jax.nn.initializers.he_uniform()

        # encoder weight matrix
        self.we = initializer(k0, (in_size, hidden_size))
        # decoder weight matrix
        self.wd = initializer(k1, (hidden_size, in_size))
        # encoder bias
        self.be = jnp.zeros((hidden_size,))
        # decader bias
        self.bd = jnp.zeros((in_size,))

    def __call__(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = ((x - self.bd) @ self.we) + self.be
        return jax.nn.relu(x)

    def decode(self, fx):
        return fx @ self.wd + self.bd

    def l1(self, x):
        x = self.encode(x)
        return jax.vmap(jnp.dot, (0, 0))(x, x)

    @staticmethod
    @filter_value_and_grad_with_aux
    def loss(diff_model, static_model, sae_pos, x, y, λ, key):
        model = eqx.combine(diff_model, static_model)
        original_activ, reconstructed_activ, pred = jax.vmap(
            lambda x, k: model(x, key=k), in_axes=(0, None), axis_name="batch"
        )(x, key)

        reconstruction_err = jnp.mean(jax.vmap(jnp.dot, (0, 0))(
            (original_activ - reconstructed_activ),
            (original_activ - reconstructed_activ)
        ))
        l1 = λ * jnp.mean(sae_pos(model).l1(original_activ))
        deep_err = jnp.mean(cross_entropy(y, pred))

        loss = reconstruction_err + l1 + deep_err
        return loss, (reconstruction_err, l1, deep_err)


def sample_features(cnn, loader, key):
    key, subkey = jax.random.split(key)
    for i, (x, _) in enumerate(loader):
        x = x.numpy()
        activ = jax.vmap(
            lambda x, k: cnn(x, key=k), in_axes=(0, None), axis_name="batch"
        )(x, subkey)[0]
        yield i, activ


def evaluate(model: eqx.Module, testloader: DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        pred_y = jnp.argmax(jax.vmap(model)(x)[-1], axis=1)
        avg_acc += jnp.mean(y == pred_y)
    return avg_acc / len(testloader)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    freeze_spec: PyTree,
    sae_pos: Callable,
    optim,
    opt_state: PyTree,
    x: Float[Array, "batch 1 28 28"],
    y: Float[Array, "batch"],
    λ: float,
    key,
):
    diff_model, static_model = eqx.partition(model, freeze_spec)
    (loss, aux), grads = SAE.loss(
        diff_model, static_model, sae_pos, x, y, λ, key
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, *aux


def train_loop(
    model: eqx.Module,
    freeze_spec: PyTree,
    sae_pos: Callable,
    trainloader: DataLoader,
    testloader: DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    tensorboard,
    λ,
    key,
) -> eqx.Module:
    opt_state = optim.init(freeze_spec)

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        key, subkey = jax.random.split(key)
        model, opt_state, loss, reconstruction_err, l1, deep_err = make_step(
            model,
            freeze_spec,
            sae_pos,
            optim,
            opt_state,
            x.numpy(),
            y.numpy(),
            λ,
            subkey,
        )
        tensorboard.add_scalar("loss", loss.item(), step)
        if (step % print_every) == 0 or (step == steps - 1):
            test_accuracy = evaluate(model, testloader)
            print(
                datetime.now().strftime("%H:%M"),
                step,
                f"{loss=:.3f}",
                f"rec={reconstruction_err.item():.3f}",
                f"{l1=:.3f}",
                f"{deep_err=:.3f}",
                f"{test_accuracy=:.3f}",
            )
            tensorboard.add_scalar("accu", test_accuracy.item(), step)
    return sae_pos(model)


def compose_model(cnn, sae, sae_pos_):
    model = sow(sae_pos_, cnn)
    model = insert_after(sae_pos_, model, sae)
    model = sow(sae_pos_, model)

    sae_pos = lambda m: sae_pos_(m).children[1]
    freeze_spec = jtu.tree_map(lambda _: False, model)
    freeze_spec = eqx.tree_at(
        sae_pos,
        freeze_spec,
        replace=jtu.tree_map(lambda leaf: eqx.is_array(leaf), sae)
    )
    
    return model, freeze_spec, sae_pos


def train_sae(
    key,
    cnn,
    sae_pos,
    activ_size,
    hidden_size,
    batch_size,
    learning_rate,
    steps,
    print_every,
    tensorboard,
    λ,
    trainloader,
    testloader
):
    key, subkey = jax.random.split(key)
    sae = SAE(activ_size, hidden_size, subkey)
    model, freeze_spec, sae_pos = compose_model(cnn, sae, sae_pos)
    optim = optax.adamw(learning_rate)
    return train_loop(
        model,
        freeze_spec,
        sae_pos,
        trainloader,
        testloader,
        optim,
        steps,
        print_every,
        tensorboard,
        λ,
        key,
    )
