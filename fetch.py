
from torch.utils.data._utils.fetch import _BaseDatasetFetcher, _MapDatasetFetcher

from PIL import Image
# from torchvision import transforms
# from transformers import BertTokenizer
import os
from typing import Union

STOCK_4000_B_PATH = os.path.join('C:/CODE/images3/stock4000_B/set')
STOCK_100_PATH = os.path.join('C:/CODE/images3/stock100/set')
FIQ_B_PATH = os.path.join('C:/CODE/images3/fiq_B/set')
ZAPPOS_PATH = os.path.join('C:/CODE/images3/zappos/set')
JEWELRY_PATH = os.path.join('C:/CODE/images3/jewelry/set')
S2_PATH = os.path.join('C:/CODE/images3/stock2000_spartoo/set/')
LAION_PATH = os.path.join('C:/CODE/images3/laion/set/')
FIQ_PATH = os.path.join('C:/CODE/images3/fiq/set/')
SCRAP_PATH = os.path.join('C:/CODE/images3/scrap/set/')
SCRAP24_PATH = os.path.join('C:/CODE/images3/scrap24/set/')
WATCHESFF_PATH = os.path.join('C:/CODE/images3/watchesFF/set/')

IMAGE_PATHS = {'s4B': STOCK_4000_B_PATH, 
               's1': STOCK_100_PATH, 
               'fiB': FIQ_B_PATH, 
               'za': ZAPPOS_PATH, 
               'sw':JEWELRY_PATH, 
               's2': S2_PATH,
               'la' : LAION_PATH,
               'fi' : FIQ_PATH,
               'scrap' : SCRAP_PATH,
               'scrap24': SCRAP24_PATH,
               'watchesFF': WATCHESFF_PATH}

def get_image_path(id : Union[str, list[str]]) -> Union[str, list[str]]:
    """Returns the path to an image given its id

    Args:
        id (Union[str, list(str)]): Image ID

    Returns:
        str: Absolute path to the image. It is None if the image is not found
    """
    if isinstance(id, list):
        paths = []
        for elt in id:
            prefix = elt.split('_')[0]
            if prefix not in IMAGE_PATHS:
                paths.append(elt)
            else:
                path = IMAGE_PATHS[prefix]
                c=0
                for t in ('.jpg', '.png'):
                    path_img = os.path.join(path, elt + t)
                    if os.path.exists(path_img):
                        paths.append(path_img)
                        break
                    c+=1
                    if c==2 : paths.append(None)
        return paths
    else:
        prefix = id.split('_')[0]
        if prefix not in IMAGE_PATHS:
            return id
        path = IMAGE_PATHS[prefix]

        for t in ('.jpg', '.png'):
            path_img = os.path.join(path, id + t)
            if os.path.exists(path_img):
                return path_img
    return None

 
def remove_dotjpg(s: str):
    if s[-4:]==".jpg":
        return(s[:-4])
    return(s)

def CV_tr(x):
    return(Image.open(get_image_path(remove_dotjpg(x))))

class _cRemoteDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset: str, auto_collation, collate_fn, drop_last, conn_str, container_name):
        super().__init__(dataset=None, auto_collation=auto_collation, collate_fn=collate_fn, drop_last=drop_last)
        self.dataset=dataset
        self.auto_collation=auto_collation
        self.collate_fn=collate_fn
        self.drop_last=drop_last
        self.conn_str=conn_str
        self.container_name=container_name
    
    def fetch(self, possibly_batched_index):
        # print("possibly_batched_index:",possibly_batched_index)
        import pandas as pd, os, shutil
        from datasets import Dataset as dDataset

        if "blob.core.windows.net" in self.dataset:
            from io import BytesIO
            from azureml._vendor.azure_storage.blob import BlobServiceClient

            if ".csv" in self.dataset:
                local_path = "C:/CODE/" + (self.dataset).split(self.container_name+"/")[1]
                if os.path.isfile(local_path):
                    # print("Found cached dataset")
                    df = pd.read_csv(local_path , sep=",", header='infer')
                else:
                    try:
                        shutil.rmtree("C:/CODE/datasetti2/"+self.dataset.split("/")[-2])
                        os.mkdir("C:/CODE/datasetti2/"+self.dataset.split("/")[-2])
                        print("Deleted cached folder")
                    except:
                        os.mkdir("C:/CODE/datasetti2/"+self.dataset.split("/")[-2])
                    preblob_client = BlobServiceClient.from_connection_string(self.conn_str)
                    blobcli = preblob_client.get_blob_client(container=self.container_name, blob=(self.dataset).split(self.container_name+"/")[1])
                    download_stream = blobcli.download_blob()
                    df = pd.read_csv(BytesIO(download_stream.readall()) , sep=",", header=0)
                    df.to_csv("C:/CODE/datasetti2/"+self.dataset.split("/")[-2]+"/"+self.dataset.split("/")[-1], index=False)
                    # print("Saved downloaded file to " + "C:/CODE/datasetti2/"+self.dataset.split("/")[-2]+"/"+self.dataset.split("/")[-1])
                self.ds = dDataset.from_pandas(df)
                try:
                    return(_MapDatasetFetcher(self.ds, self.auto_collation, self.collate_fn, self.drop_last).fetch(possibly_batched_index))
                except IndexError:
                    raise StopIteration

        elif ".csv" in self.dataset:
            import requests as r
            from io import StringIO
            response = r.get(self.dataset)
            response_text = response.text
            if response.status_code >= 400:
                raise "Could not retrieve data from provided URL"
            df = pd.read_csv(StringIO(response_text), sep=",", header='infer')
            
            cols = df.columns
            for c in cols:
                if "image" in c:
                    df[c] = pd.Series([CV_tr(i) for i in df[c]])

            self.ds = dDataset.from_pandas(df)
            return(_MapDatasetFetcher(self.ds, self.auto_collation, self.collate_fn, self.drop_last).fetch(possibly_batched_index))

        else:
            raise Exception("Provided URL is not a remote .csv file")
