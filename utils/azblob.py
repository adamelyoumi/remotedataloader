
import pandas as pd
from io import BytesIO
import os
import shutil
from azureml._vendor.azure_storage.blob import BlobServiceClient

conn_str = 'my_connection_string'
container_name = 'my_container_name'

def download_easy(remotepath:str, conn_str:str=conn_str, container_name:str=container_name) -> pd.DataFrame:
    preblob_client = BlobServiceClient.from_connection_string(conn_str)
    blobcli = preblob_client.get_blob_client(container=container_name, blob=remotepath)
    download_stream = blobcli.download_blob()
    df = pd.read_csv(BytesIO(download_stream.readall()) , sep=",", header=0)
    return(df)

def upload_easy(localpath:str, remotepath:str, conn_str:str=conn_str, container_name:str=container_name, overwrite:bool=True):
    preblob_client = BlobServiceClient.from_connection_string(conn_str)
    blobcli = preblob_client.get_blob_client(container=container_name, blob=remotepath)
    with open(localpath, "rb") as data:
        blobcli.upload_blob(data, overwrite=overwrite)

def partition_csv(file:str, batch_size:int, overwrite:bool = False):
    folder = '/'.join(file.split("/")[:-1])
    filename =  file.split("/")[-1].removesuffix(".csv")
    partition_folder = folder + f"/partition_{filename}/"

    if f"partition_{filename}" in os.listdir(folder) and not overwrite:
        print("Found existing partition")
        return()

    elif f"partition_{filename}" in os.listdir(folder) and overwrite:
        print("Overwriting existing partition")
        shutil.rmtree(partition_folder)
        os.mkdir(partition_folder)
    
    elif f"partition_{filename}" not in os.listdir(folder):
        print("No partition found")
        os.mkdir(partition_folder)

    print("Creating partition...")
    with open(file, "r", encoding="utf8") as f:
        len_file = len(f.readlines())

    df_iter = pd.read_csv(file, skiprows = 0, chunksize = batch_size)
    
    for i in range(int(len_file/batch_size)+1):
        df = next(df_iter)
        try:
            df.drop('Unnamed: 0', axis=1).to_csv(partition_folder + filename + f"_{i}.csv", index=False)
        except:
            df.to_csv(partition_folder + filename + f"_{i}.csv", index=False)

def upload_partition(file, batch_size:int, target_path=None, clean_local:bool=True, overwrite:bool = False):
    """
    Creates Partition for provided file and uploads it to the specified path on the blob

    Args:
        file (str): path to local .csv file
        batch_size (int): The size of each partition bit
        target_path (str, optional): Target path on the blob. Defaults to "/datasets/partition_{filename}/".
        clean_local (bool, optional): Whether to clean the partition generated locally. Defaults to True.
    """

    folder = '/'.join(file.split("/")[:-1])
    filename =  file.split("/")[-1].removesuffix(".csv")
    partition_folder = folder + f"/partition_{filename}/"

    if target_path is None:
        target_path = "datasets/" + f"partition_{filename}/"

    partition_csv(file, batch_size=batch_size, overwrite=overwrite)

    print("Starting upload...")
    k=0
    for file in os.listdir(partition_folder):
        upload_easy(os.path.join(partition_folder,file), target_path + filename + f"_{k}.csv")
        print(f"Uploaded file {os.path.join(partition_folder, file)} to blob")
        k+=1

    if clean_local:
        shutil.rmtree(partition_folder)


if __name__=="__main__":

    upload_partition("path/to/file.csv", batch_size = 256, overwrite = True)
