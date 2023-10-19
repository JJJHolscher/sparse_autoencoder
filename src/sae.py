#! /usr/bin/env python3
# vim:fenc=utf-8

"""

"""

import equinox as eqx
import jax
import jax.numpy as jnp
import mnist
import optax
from jaxtyping import Array, Float, Int, PyTree
from jo3util.eqx import sow
from torch.utils.data import DataLoader


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

    def encode(self, x):
        x = (x - self.bd) @ self.we + self.be
        return jax.nn.relu(x)

    def decode(self, fx):
        return fx @ self.wd + self.bd

    @staticmethod
    @eqx.filter_value_and_grad
    def loss(sae, x, λ):
        fx = jax.vmap(sae.encode)(x)
        x_ = jax.vmap(sae.decode)(fx)
        sq_err = jnp.dot((x - x_), (x - x_))
        l1 = λ * jnp.dot(fx, fx)
        return jnp.mean(sq_err + l1)


def train_loop(
    sae: SAE,
    model: eqx.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> eqx.Module:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        sae: SAE,
        model: eqx.Module,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        _, activ = jax.vmap(model)(x)
        loss_value, grads = SAE.loss(sae, activ[0], 1)
        updates, opt_state = optim.update(grads, opt_state, sae)
        sae = eqx.apply_updates(sae, updates)
        return sae, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        sae, opt_state, train_loss = make_step(sae, model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(sae, model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model


def train_sae(
    key,
    cnn,
    sae_pos,
    activ_size,
    batch_size,
    learning_rate,
    steps,
    print_every,
    sae_storage="./res/sae.eqx",
):
    trainloader, testloader = mnist.load(batch_size=batch_size)
    model = sow(sae_pos, cnn)
    print(model)
    sae = SAE(activ_size, 1000, key)
    optim = optax.adamw(learning_rate)
    model = train_loop(
        sae, model, trainloader, testloader, optim, steps, print_every
    )
    # eqx.tree_serialise_leaves(model_storage, model)
    return model
