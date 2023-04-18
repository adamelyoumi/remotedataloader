from torch.utils.data.sampler import Sampler
from typing import *

class cSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, length: int = None) -> None:
        self.data_source = data_source
        self.length = length

    def __iter__(self) -> Iterator[int]:
        return iter(range((len(self.data_source) if self.length is None else self.length)))

    def __len__(self) -> int:
        return(len(self.data_source) if self.length is None else self.length)