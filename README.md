# Remote Pytorch DataLoader

## Overview
New RemoteDataLoader class to support URL input.
The class curretly handles URLs that provide unrestricted access to a .csv file, as well as .csv files hosted on an Azure Blob storage Instance.
To access your blob, specify the connection string and container name in the DataLoader initialization parameters.


## General idea
This class was originally designed to handle huge datasets which would be too costly to download at every new training task.
The new class was designed with the idea of only downloading the data effectively used for training. As a result, it is best used with a remote dataset partitioned into same-size .csv files
If it is not found in the cache, the file is downloaded when the first batch is retrieved from the dataloader's iterator

This repo also provides utilities to create the partitions and upload them to an Azure Blob storage instance.

```python
remote_ds = f"https://my_storage_account.blob.core.windows.net/my_container_name/path_to_partition/partition_{dataset_name}/{dataset_name}_{file_number}.csv" # Partition file of length 48
connect_str = "my_connection_string"
container_name = "container_name"
remote_dataloader = RemoteDataLoader(remote_ds, connect_str=connect_str, container_name=container_name, batch_size=16, file_length=48)

iterator = iter(remote_dataloader)
next(iterator) # Downloads the file, caches it, and returns items 0-15 (the first batch)
next(iterator) # Finds the cached file, returns items 16-31
next(iterator) # Finds the cached file, returns items 32-47
next(iterator) # Raises StopIteration
```

## Changes
The modifications compared to Pytorch's DataLoader are the following:

- ``torch/utils/data/dataloader.py`` : Defines the RemoteDataLoader class, inheriting from pytorch's DataLoader

- ``torch/utils/data/sampler.py`` : Defines the sampler length in the case of a remote dataset

- ``torch/utils/data/_utils/fetch.py`` : Defines a ``RemoteDatasetFetcher`` class that handles the file download and redirects to ``MapDatasetFetcher.fetch()``

## Limitations

- Currently only handles a single worker