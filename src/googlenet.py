#! /usr/bin/env python3

import logging
import os
import sys
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import torch
from jaxtyping import Array


def load_torch_weights(
    model: eqx.Module,
    state: Optional[nn.State],
    torch_weights: Optional[str] = None,
    tmp_dir=".eqx",
) -> Tuple[eqx.Module, Optional[nn.State]]:
    """Loads weights from a PyTorch serialised file.

    ???+ warning

        - This method requires installation of the [`torch`](https://pypi.org/project/torch/) package.

    !!! note

        - This function assumes that Eqxvision's ordering of class
          attributes mirrors the `torchvision.models` implementation.
        - This method assumes the `eqxvision` model is *not* initialised.
            Problems arise due to initialised `BN` modules.
        - The saved checkpoint should **only** contain model parameters as keys.

    !!! info
        A full list of pretrained URLs is provided
        [here](https://github.com/paganpasta/eqxvision/blob/main/eqxvision/utils.py).

    **Arguments:**

    - `model`: An `eqx.Module` for which the `jnp.ndarray` leaves are
        replaced by corresponding `PyTorch` weights.
    - `torch_weights`: A string either pointing to `PyTorch` weights on disk or the download `URL`.

    **Returns:**
        The model with weights loaded from the `PyTorch` checkpoint.
    """
    if "torch" not in sys.modules:
        raise RuntimeError(
            " Torch package not found! Pretrained is only supported with the torch package."
        )

    if torch_weights is None:
        raise ValueError("torch_weights parameter cannot be empty!")

    if not os.path.exists(torch_weights):
        global _TEMP_DIR
        filepath = os.path.join(tmp_dir, os.path.basename(torch_weights))
        if os.path.exists(filepath):
            logging.info(
                f"Downloaded file exists at f{filepath}. Using the cached file!"
            )
        else:
            os.makedirs(tmp_dir, exist_ok=True)
            torch.hub.download_url_to_file(torch_weights, filepath)
    else:
        filepath = torch_weights
    saved_weights = torch.load(filepath, map_location="cpu")
    weights_iterator = iter(
        [
            (name, jnp.asarray(weight.detach().numpy()))
            for name, weight in saved_weights.items()
            if "num_batches" not in name and "running" not in name
        ]
    )

    running_mean = None
    bn_s = []
    for name, weight in saved_weights.items():
        if "running_mean" in name:
            bn_s.append(False)
            assert running_mean is None
            running_mean = jnp.asarray(weight.detach().numpy())
        elif "running_var" in name:
            assert running_mean is not None
            bn_s.append((running_mean, jnp.asarray(weight.detach().numpy())))
            running_mean = None
    bn_iterator = iter(bn_s)

    leaves, tree_def = jtu.tree_flatten(model)

    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and not (
            leaf.size == 1 and isinstance(leaf.item(), bool)
        ):
            (weight_name, new_weights) = next(weights_iterator)
            new_leaves.append(jnp.reshape(new_weights, leaf.shape))
        else:
            new_leaves.append(leaf)

    model = jtu.tree_unflatten(tree_def, new_leaves)

    for state_index in jtu.tree_leaves(
        model, is_leaf=lambda m: isinstance(m, nn.StateIndex)
    ):
        if not isinstance(state_index, nn.StateIndex):
            continue
        if state is None:
            raise ValueError("provide an initial state for stateful models")

        state = state.set(state_index, next(bn_iterator))

    # def set_experimental(iter_bn, x):
    # def set_values(y):
    # if isinstance(y, nn.StateIndex):
    # current_val = next(iter_bn)
    # if isinstance(current_val, bool):
    # eqx.experimental.set_state(y, jnp.asarray(False))
    # else:
    # running_mean, running_var = current_val, next(iter_bn)
    # eqx.experimental.set_state(y, (running_mean, running_var))
    # return y

    # return jtu.tree_map(
    # set_values, x, is_leaf=lambda _: isinstance(_, nn.StateIndex)
    # )

    # model = jtu.tree_map(set_experimental, bn_iterator, model)
    return model, state


class BasicConv2d(nn.StatefulLayer):
    conv: nn.Conv2d
    bn: nn.BatchNorm

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: Optional[jax.random.PRNGKeyArray] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if key is None:
            self.conv = nn.Conv2d(
                in_channels, out_channels, use_bias=False, **kwargs
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, use_bias=False, key=key, **kwargs
            )
        self.bn = nn.BatchNorm(out_channels, axis_name="batch", eps=0.001)

    def __call__(
        self, x: Array, state: nn.State, key: jax.random.PRNGKeyArray
    ) -> Tuple[Array, nn.State]:
        x = self.conv(x)
        x, state = self.bn(x, state, key=key)
        return jnn.relu(x), state


class InceptionAux(eqx.Module):
    conv: BasicConv2d
    fc1: nn.Linear
    fc2: nn.Linear
    dropout: nn.Dropout
    avgpool: nn.AdaptiveAvgPool2d

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Callable = BasicConv2d,
        dropout: float = 0.7,
        *,
        key: jax.random.PRNGKeyArray,
    ) -> None:
        super().__init__()
        keys = jrandom.split(key, 3)
        self.conv = conv_block(in_channels, 128, kernel_size=1, key=keys[0])
        self.fc1 = nn.Linear(2048, 1024, key=keys[1])
        self.fc2 = nn.Linear(1024, num_classes, key=keys[2])
        self.dropout = nn.Dropout(p=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (4, 4),
        )

    def __call__(
        self, x: Array, state: nn.State, *, key: jax.random.PRNGKeyArray
    ) -> Tuple[Array, nn.State]:
        keys = jrandom.split(key, 2)
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.avgpool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x, state = self.conv(x, state, key=keys[0])
        # N x 128 x 4 x 4
        x = jnp.ravel(x)
        # N x 2048
        x = jnn.relu(self.fc1(x))
        # N x 1024
        x = self.dropout(x, key=keys[1])
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x, state


class _Inception(eqx.Module):
    branch1: BasicConv2d
    branch2: nn.Sequential
    branch3: nn.Sequential
    branch4: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Callable = BasicConv2d,
        *,
        key: jax.random.PRNGKeyArray,
    ) -> None:
        super().__init__()
        keys = jrandom.split(key, 5)
        self.branch1 = conv_block(
            in_channels, ch1x1, kernel_size=1, key=keys[0]
        )
        self.branch2 = nn.Sequential(
            [
                conv_block(in_channels, ch3x3red, kernel_size=1, key=keys[1]),
                conv_block(
                    ch3x3red, ch3x3, kernel_size=3, padding=1, key=keys[2]
                ),
            ]
        )

        self.branch3 = nn.Sequential(
            [
                conv_block(in_channels, ch5x5red, kernel_size=1, key=keys[3]),
                # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
                # Please see https://github.com/pytorch/vision/issues/906 for details.
                conv_block(
                    ch5x5red, ch5x5, kernel_size=3, padding=1, key=keys[4]
                ),
            ]
        )

        self.branch4 = nn.Sequential(
            [
                nn.MaxPool2d(
                    kernel_size=3, stride=1, padding=1, use_ceil=True
                ),
                conv_block(in_channels, pool_proj, kernel_size=1, key=keys[5]),
            ]
        )

    def __call__(
        self, x: Array, state: nn.State, *, key: jax.random.PRNGKeyArray
    ) -> Tuple[Array, nn.State]:
        keys = jrandom.split(key, 4)
        branch1, state = self.branch1(x, state, key=keys[0])
        branch2, state = self.branch2(x, state, key=keys[1])
        branch3, state = self.branch3(x, state, key=keys[2])
        branch4, state = self.branch4(x, state, key=keys[3])

        outputs = jnp.concatenate([branch1, branch2, branch3, branch4], axis=0)
        return outputs, state


class GoogLeNet(eqx.Module):
    """A simple port of `torchvision.models.GoogLeNet`"""

    aux_logits: bool
    conv1: BasicConv2d
    maxpool1: nn.MaxPool2d
    conv2: BasicConv2d
    conv3: BasicConv2d
    maxpool2: nn.MaxPool2d
    inception3a: _Inception
    inception3b: _Inception
    maxpool3: nn.MaxPool2d
    inception4a: _Inception
    inception4b: _Inception
    inception4c: _Inception
    inception4d: _Inception
    inception4e: _Inception
    maxpool4: nn.MaxPool2d
    inception5a: _Inception
    inception5b: _Inception
    aux1: Optional[InceptionAux]
    aux2: Optional[InceptionAux]
    avgpool: nn.AdaptiveAvgPool2d
    dropout: nn.Dropout
    fc: nn.Linear

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = False,
        blocks: List[Callable] = [BasicConv2d, _Inception, InceptionAux],
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        *,
        key: jax.random.PRNGKeyArray = None,
    ) -> None:
        """
        **Arguments:**

        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - `aux_logits`: If `True`, two auxiliary branches are added to the network. Defaults to `False`
        - `blocks`: Blocks for constructing the network
        - `dropout`: Dropout applied on the `main` branch. Defaults to `0.2`
        - `dropout_aux`: Dropout applied on the `aux` branches. Defaults to `0.7`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        """
        super().__init__()
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 20)

        self.aux_logits = aux_logits
        self.conv1 = conv_block(
            3, 64, kernel_size=7, stride=2, padding=3, key=keys[0]
        )
        self.maxpool1 = nn.MaxPool2d(3, stride=2, use_ceil=True)
        self.conv2 = conv_block(64, 64, kernel_size=1, key=keys[1])
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1, key=keys[2])
        self.maxpool2 = nn.MaxPool2d(3, stride=2, use_ceil=True)

        self.inception3a = inception_block(
            192, 64, 96, 128, 16, 32, 32, key=keys[3]
        )
        self.inception3b = inception_block(
            256, 128, 128, 192, 32, 96, 64, key=keys[4]
        )
        self.maxpool3 = nn.MaxPool2d(3, stride=2, use_ceil=True)

        self.inception4a = inception_block(
            480, 192, 96, 208, 16, 48, 64, key=keys[5]
        )
        self.inception4b = inception_block(
            512, 160, 112, 224, 24, 64, 64, key=keys[6]
        )
        self.inception4c = inception_block(
            512, 128, 128, 256, 24, 64, 64, key=keys[7]
        )
        self.inception4d = inception_block(
            512, 112, 144, 288, 32, 64, 64, key=keys[8]
        )
        self.inception4e = inception_block(
            528, 256, 160, 320, 32, 128, 128, key=keys[9]
        )
        self.maxpool4 = nn.MaxPool2d(2, stride=2, use_ceil=True)

        self.inception5a = inception_block(
            832, 256, 160, 320, 32, 128, 128, key=keys[10]
        )
        self.inception5b = inception_block(
            832, 384, 192, 384, 48, 128, 128, key=keys[11]
        )

        self.aux1 = None
        self.aux2 = None
        if aux_logits:
            self.aux1 = inception_aux_block(
                512, num_classes, dropout=dropout_aux, key=keys[12]
            )
            self.aux2 = inception_aux_block(
                528, num_classes, dropout=dropout_aux, key=keys[13]
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes, key=keys[14])

    def __call__(
        self, x: Array, state: nn.State, *, key: jax.random.PRNGKeyArray
    ) -> Tuple[Union[Array, Optional[Array], Optional[Array]], nn.State]:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, 14)
        # N x 3 x 224 x 224
        x, state = self.conv1(x, state, key=keys[0])
        # N x 64 x 112 x 112
        x = self.maxpool1(x, key=keys[1])
        # N x 64 x 56 x 56
        x, state = self.conv2(x, state, key=keys[2])
        # N x 64 x 56 x 56
        x, state = self.conv3(x, state, key=keys[3])
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x, state = self.inception3a(x, state, key=keys[4])
        # N x 256 x 28 x 28
        x, state = self.inception3b(x, state, key=keys[5])
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x, state = self.inception4a(x, state, key=keys[6])
        # N x 512 x 14 x 14
        if self.aux_logits:
            aux1, state = self.aux1(x, state, key=keys[7])

        x, state = self.inception4b(x, state, key=keys[8])
        # N x 512 x 14 x 14
        x, state = self.inception4c(x, state, key=keys[9])
        # N x 512 x 14 x 14
        x, state = self.inception4d(x, state, key=keys[10])
        # N x 528 x 14 x 14
        if self.aux_logits:
            aux2, state = self.aux2(
                x, state, key=keys[11]
            )  # Key here, a bad thing?

        x, state = self.inception4e(x, state, key=keys[12])
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x, state = self.inception5a(x, state, key=keys[13])
        # N x 832 x 7 x 7
        x, state = self.inception5b(x, state, key=keys[14])
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = jnp.ravel(x)
        # N x 1024
        x = self.dropout(x, key=keys[15])
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.aux_logits:
            return x, aux2, aux1, state
        else:
            return x, state


def googlenet(
    torch_weights: Optional[str] = None, **kwargs: Any
) -> Tuple[GoogLeNet, nn.State]:
    r"""GoogLeNet (Inception v1) model architecture from
    [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842).
    The required minimum input size of the model is 15x15.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`

    """
    if torch_weights:
        use_aux = kwargs.get("aux_logits", False)
        model, state = nn.make_with_state(GoogLeNet)(aux_logits=True, **kwargs)
        model, state = load_torch_weights(
            model, state, torch_weights=torch_weights
        )
        if not use_aux:
            model = eqx.tree_at(lambda m: m.aux_logits, model, replace=(False))
        else:
            warnings.warn(
                "Loaded torch_weights weights for GoogLeNet. But, aux-branch weights are un-trained."
            )
    else:
        model, state = nn.make_with_state(GoogLeNet)(**kwargs)
    return model, state
