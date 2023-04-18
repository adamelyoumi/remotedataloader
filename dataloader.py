
import torch
import torch.utils.data.dataloader as d
from torch.utils.data import _utils
from typing import *

from remotedataloader.fetch import _cRemoteDatasetFetcher
from remotedataloader.sampler import cSequentialSampler

class _cDatasetKind(d._DatasetKind):
    Map = 0
    Iterable = 1
    Remote = 2

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last, conn_str, container_name):
        if kind == d._DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        elif kind == d._DatasetKind.Iterable:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _cRemoteDatasetFetcher(dataset, auto_collation, collate_fn, drop_last, conn_str, container_name)


class RemoteDataLoader(d.DataLoader):
    def __init__(self, 
                 dataset: Union[d.Dataset[d.T_co], str], 
                 batch_size: Optional[int] = 1, 
                 shuffle: Optional[bool] = None, 
                 sampler: Union[d.Sampler, Iterable, None] = None, 
                 batch_sampler: Union[d.Sampler[Sequence], Iterable[Sequence], None] = None, 
                 num_workers: int = 0, collate_fn: Optional[d._collate_fn_t] = None, 
                 pin_memory: bool = False, drop_last: bool = False, 
                 timeout: float = 0, worker_init_fn: Optional[d._worker_init_fn_t] = None, 
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2, 
                 persistent_workers: bool = False, pin_memory_device: str = "",
                 connect_str: str = None, container_name: str = None, file_length: int = None
                 ):
        if isinstance(dataset, d.Dataset):
            super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor, persistent_workers, pin_memory_device)
        
        else:
            torch._C._log_api_usage_once("python.data_loader")
            if num_workers < 0:
                raise ValueError('num_workers option should be non-negative; '
                                'use num_workers=0 to disable multiprocessing.')

            if timeout < 0:
                raise ValueError('timeout option should be non-negative')

            if num_workers == 0 and prefetch_factor != 2:
                raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                                'let num_workers > 0 to enable multiprocessing.')
            assert prefetch_factor > 0

            if persistent_workers and num_workers == 0:
                raise ValueError('persistent_workers option needs num_workers > 0')

            self.dataset = dataset
            self.num_workers = num_workers
            self.prefetch_factor = prefetch_factor
            self.pin_memory = pin_memory
            self.pin_memory_device = pin_memory_device
            self.timeout = timeout
            self.worker_init_fn = worker_init_fn
            self.multiprocessing_context = multiprocessing_context
            self.connect_str = connect_str
            self.container_name = container_name
            self.file_length = file_length if file_length is not None else batch_size

            if isinstance(self.dataset, d.IterDataPipe):
                self.dataset = d._IterDataPipeSerializationWrapper(self.dataset)
            elif isinstance(self.dataset, d.MapDataPipe):
                self.dataset = d._MapDataPipeSerializationWrapper(self.dataset)

            if isinstance(dataset, d.IterableDataset):
                self._dataset_kind = _cDatasetKind.Iterable
                if isinstance(dataset, d.IterDataPipe):
                    if shuffle is not None:
                        dataset = torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
                # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
                elif shuffle not in {False, None}:
                    raise ValueError(
                        "DataLoader with IterableDataset: expected unspecified "
                        "shuffle option, but got shuffle={}".format(shuffle))

                if sampler is not None:
                    # See NOTE [ Custom Samplers and IterableDataset ]
                    raise ValueError(
                        "DataLoader with IterableDataset: expected unspecified "
                        "sampler option, but got sampler={}".format(sampler))
                elif batch_sampler is not None:
                    # See NOTE [ Custom Samplers and IterableDataset ]
                    raise ValueError(
                        "DataLoader with IterableDataset: expected unspecified "
                        "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
            elif isinstance(dataset, str):
                self._dataset_kind = _cDatasetKind.Remote
            else:
                shuffle = bool(shuffle)
                self._dataset_kind = _cDatasetKind.Map


            if sampler is not None and shuffle:
                raise ValueError('sampler option is mutually exclusive with '
                                'shuffle')

            if batch_sampler is not None:
                # auto_collation with custom batch_sampler
                if batch_size != 1 or shuffle or sampler is not None or drop_last:
                    raise ValueError('batch_sampler option is mutually exclusive '
                                    'with batch_size, shuffle, sampler, and '
                                    'drop_last')
                batch_size = None
                drop_last = False
            elif batch_size is None:
                # no auto_collation
                if drop_last:
                    raise ValueError('batch_size=None option disables auto-batching '
                                    'and is mutually exclusive with drop_last')

            if sampler is None:  # give default samplers
                if self._dataset_kind == _cDatasetKind.Iterable:
                    # See NOTE [ Custom Samplers and IterableDataset ]
                    sampler = d._InfiniteConstantSampler()
                elif self._dataset_kind == _cDatasetKind.Remote:
                    sampler = cSequentialSampler(dataset, self.file_length)
                else: #self._dataset_kind == _DatasetKind.Map:
                    if shuffle:
                        sampler = d.RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                    else:
                        sampler = d.SequentialSampler(dataset)  # type: ignore[arg-type]
                # else:
                #     raise NotImplementedError

            if batch_size is not None and batch_sampler is None:
                # auto_collation without custom batch_sampler
                batch_sampler = d.BatchSampler(sampler, batch_size, drop_last)

            self.batch_size = batch_size
            self.drop_last = drop_last
            self.sampler = sampler
            self.batch_sampler = batch_sampler
            self.generator = generator

            if collate_fn is None:
                if self._auto_collation:
                    collate_fn = _utils.collate.default_collate
                else:
                    collate_fn = _utils.collate.default_convert

            self.collate_fn = collate_fn
            self.persistent_workers = persistent_workers

            self.__initialized = True
            self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

            self._iterator = None

            self.check_worker_number_rationality()

            torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

        
    def _get_iterator(self) -> 'd._BaseDataLoaderIter':
        if self.num_workers == 0:
            return _cSingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return d._MultiProcessingDataLoaderIter(self)


class _cSingleProcessDataLoaderIter(d._BaseDataLoaderIter):
    def __init__(self, loader):
        super(_cSingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._conn_str = loader.connect_str
        self._container_name = loader.container_name

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (d.IterDataPipe, d.MapDataPipe)):
            torch.utils.data.graph_settings.apply_sharding(self._dataset, self._world_size, self._rank)

        self._dataset_fetcher = _cDatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last, self._conn_str, self._container_name)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data
