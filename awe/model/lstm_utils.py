from typing import Callable
import torch

def apply_to_pack(
    pack: torch.nn.utils.rnn.PackedSequence,
    fn: Callable[[torch.Tensor], torch.Tensor]
):
    return re_pack(pack, fn(pack.data))

def re_pack(pack: torch.nn.utils.rnn.PackedSequence, data: torch.Tensor):
    # Inspired by
    # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4.
    return torch.nn.utils.rnn.PackedSequence(
        data=data,
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )

# From https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/8.

def sorted_lengths(pack: torch.nn.utils.rnn.PackedSequence):
    indices = torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )
    lengths = ((indices + 1)[:, None] <= pack.batch_sizes[None, :]).long().sum(dim=1)
    return lengths, indices

def sorted_first_indices(pack: torch.nn.utils.rnn.PackedSequence):
    return torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )

def sorted_last_indices(pack: torch.nn.utils.rnn.PackedSequence):
    lengths, indices = sorted_lengths(pack)
    cum_batch_sizes = torch.cat([
        pack.batch_sizes.new_zeros((2,)),
        torch.cumsum(pack.batch_sizes, dim=0),
    ], dim=0)
    return cum_batch_sizes[lengths] + indices

def first_items(pack: torch.nn.utils.rnn.PackedSequence, unsort: bool) -> torch.Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.data[pack.unsorted_indices]
    return pack.data[:pack.batch_sizes[0]]

def last_items(pack: torch.nn.utils.rnn.PackedSequence, unsort: bool) -> torch.Tensor:
    indices = sorted_last_indices(pack=pack)
    if unsort and pack.unsorted_indices is not None:
        indices = indices[pack.unsorted_indices]
    return pack.data[indices]
