# from clint.textui import progress
import requests
from tqdm import tqdm
import zipfile
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'  # https -> http
}


def get_data_from_dataset(dataset: str, zip_path):
    horse2zebra_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'
    res = requests.get(horse2zebra_url, stream=True, proxies=proxies)
    total_length = int(res.headers.get('content-length'))
    chunk_size = 1024 * 1024

    file_name = "./%s.zip" % dataset
    is_download = True
    is_extract = True
    if os.path.exists(file_name):
        now_size = os.path.getsize(file_name)
        is_download = False
        if now_size != total_length:
            is_download = True
        if is_download:
            print("now downloading ")
        else:
            print("had downloaded")
    if is_download:
        with open(file_name, "wb") as zip_file:
            for chunk in tqdm(iterable=res.iter_content(chunk_size=chunk_size), total=total_length / chunk_size,
                              desc=dataset, unit='MB'):
                # print("write i")
                zip_file.write(chunk)
    if not os.path.exists(zip_path):
        os.makedirs(zip_path)
    else:
        print('path exist')
    full_name = zip_path + '/' + dataset
    if os.path.exists(full_name):
        # print("file ex")
        is_extract = False
    # print(full_name)
    if is_extract:
        zf = zipfile.ZipFile(file_name, mode='r')
        for f_n in zf.namelist():
            zf.extract(f_n, path=zip_path)
        zf.close()
        print("extract finished ")


class ImageDataSet(Dataset):
    def __init__(self, dataset: str, zip_path: str, transforms_=None):
        self.transforms = torchvision.transforms.Compose(transforms_)
        # zip_path='./data'
        get_data_from_dataset(dataset, zip_path)
        # self.files_testA = glob.glob(os.path.join(zip_path, dataset, 'testA') + '/*.*')
        # self.files_testB = glob.glob(os.path.join(zip_path, dataset, 'testB') + '/*.*')
        self.files_trainA = glob.glob(os.path.join(zip_path, dataset, 'trainA') + '/*.*')
        self.files_trainB = glob.glob(os.path.join(zip_path, dataset, 'trainB') + '/*.*')
        self.len_trainA = len(self.files_trainA)
        self.len_trainB = len(self.files_trainB)
        # print(self.files_trainA)
        pass

    def __getitem__(self, index):
        image_trainA = Image.open(self.files_trainA[index % self.len_trainA])
        image_trainB = Image.open(self.files_trainB[index % self.len_trainB])

        item_A = self.transforms(image_trainA)
        item_B = self.transforms(image_trainB)

        return {
            'A': item_A,
            'B': item_B,
        }

        pass

    def __len__(self):
        return max(self.len_trainA, self.len_trainB)


# %%
z_path = './data'
batch_size = 4
trans = torchvision.transforms.ToTensor()
horse_zebra_dataset = ImageDataSet(dataset='horse2zebra', zip_path=z_path, transforms_=trans)
train_loader = DataLoader(horse_zebra_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          drop_last=True)
# for bat in train_loader:
#     # print(t_A)
#     # len(t_A)
#     print("nnn")
#     break

print(train_loader)
