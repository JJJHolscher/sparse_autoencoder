#! /usr/bin/env python3
# vim:fenc=utf-8

"""

"""

from datetime import datetime
from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jo3mnist
import optax
from jaxtyping import Array, Float, Int, PyTree
from jo3util.eqx import insert_after, sow
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class SAE(eqx.Module):
    we: Float
    wd: Float
    be: Float
    bd: Float

    def __init__(self, in_size, hidden_size, key=jax.random.PRNGKey(42)):
        k0, k1, k2, k3 = jax.random.split(key, 4)

        # encoder weight matrix
        self.we = jax.random.uniform(k0, (in_size, hidden_size))
        # decoder weight matrix
        self.wd = jax.random.uniform(k1, (hidden_size, in_size))
        # encoder bias
        self.be = jax.random.uniform(k2, (hidden_size,))
        # decader bias
        self.bd = jax.random.uniform(k3, (in_size,))

    def __call__(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = ((x - self.bd) @ self.we) + self.be
        return jax.nn.relu(x)

    def decode(self, fx):
        return fx @ self.wd + self.bd

    @staticmethod
    @eqx.filter_value_and_grad
    def loss(sae, x, λ):
        fx = jax.vmap(sae.encode)(x)
        x_ = jax.vmap(sae.decode)(fx)
        sq_err = jax.vmap(jnp.dot, (0, 0))((x - x_), (x - x_))
        l1 = λ * jax.vmap(jnp.dot, (0, 0))(fx, fx)
        out = jnp.mean(sq_err + l1)
        return out


def sample_features(cnn, sae, loader):
    for i, (x, _) in enumerate(loader):
        x = x.numpy()
        activ = jax.vmap(cnn)(x)[0]
        yield i, sae.encode(activ)


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
    sae: SAE,
    model: eqx.Module,
    optim,
    opt_state: PyTree,
    x: Float[Array, "batch 1 28 28"],
    λ: float,
):
    activ = jax.vmap(model)(x)[0]
    loss_value, grads = SAE.loss(sae, activ[0], λ)
    updates, opt_state = optim.update(grads, opt_state, sae)
    sae = eqx.apply_updates(sae, updates)
    return sae, opt_state, loss_value


def train_loop(
    sae: SAE,
    cnn: eqx.Module,
    sae_pos: Callable,
    trainloader: DataLoader,
    testloader: DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    tensorboard,
    λ,
) -> eqx.Module:
    opt_state = optim.init(eqx.filter(sae, eqx.is_array))

    cnn = sow(sae_pos, cnn)

    print(f"test_accuracy={evaluate(cnn, testloader).item()}")

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, _) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        sae, opt_state, train_loss = make_step(
            sae, cnn, optim, opt_state, x.numpy(), λ
        )
        tensorboard.add_scalar("loss", train_loss.item(), step)
        if (step % print_every) == 0 or (step == steps - 1):
            cnn_with_sae = insert_after(sae_pos, cnn, sae)
            test_accuracy = evaluate(cnn_with_sae, testloader)
            print(
                datetime.now().strftime("%H:%M"),
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_accuracy={test_accuracy.item()}",
            )
            tensorboard.add_scalar("accu", test_accuracy.item(), step)
    return sae


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
):
    trainloader, testloader = jo3mnist.load(batch_size=batch_size)
    sae = SAE(activ_size, hidden_size, key)
    optim = optax.adamw(learning_rate)
    return train_loop(
        sae,
        cnn,
        sae_pos,
        trainloader,
        testloader,
        optim,
        steps,
        print_every,
        tensorboard,
        λ
    )
