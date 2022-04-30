"""Utilities for working with LSTMs in PyTorch."""

from typing import Callable

import torch


def apply_to_pack(
    pack: torch.nn.utils.rnn.PackedSequence,
    fn: Callable[[torch.Tensor], torch.Tensor]
):
    """Applies `fn` transformation to data inside `pack`."""

    return re_pack(pack, fn(pack.data))

def re_pack(pack: torch.nn.utils.rnn.PackedSequence, data: torch.Tensor):
    """Packs `data` in the same way as the original `pack`."""

    # Inspired by
    # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4.
    return torch.nn.utils.rnn.PackedSequence(
        data=data,
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
