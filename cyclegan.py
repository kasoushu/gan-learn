import glob
import itertools
import os
import os.path
import time
import zipfile
import torch.nn.functional as F
import requests
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

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
        image_trainA = Image.open(self.files_trainA[index % self.len_trainA]).convert('RGB')
        image_trainB = Image.open(self.files_trainB[index % self.len_trainB]).convert('RGB')

        return self.transforms(image_trainA), self.transforms(image_trainB)

        pass

    def __len__(self):
        return max(self.len_trainA, self.len_trainB)


# model


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # x =  self.model(x)
        for model in self.model:
            # print(x.shape)
            x = model(x)
        # Average pooling and flatten
        # print(x.size())
        # print("aaa",x.size()[0])
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


## train
is_cuda = torch.cuda.is_available()
input_channel, output_channel = 3, 3
epochs = 2
now_epoch = 0
batch_size = 1
z_path = './data'
decay_epoch = epochs // 2
height, width = 256, 256
lr = 0.0002
n_works = 4
device = torch.device('cuda' if is_cuda else 'cpu')
img_path = "img/cyclegan"
# read training data


G_AB = Generator(input_channel, output_channel, n_residual_blocks=3)
G_BA = Generator(output_channel, input_channel, n_residual_blocks=3)
D_A = Discriminator(input_channel)
D_B = Discriminator(output_channel)

try:
    save_info = torch.load("cyclegan.pkl")
except Exception:
    # exit()
    pass
else:
    now_epoch = save_info["epoch"]
    G_AB.model.load_state_dict(save_info['g_AB_state'])
    G_BA.model.load_state_dict(save_info['g_BA_state'])
    D_A.model.load_state_dict(save_info['D_A_state'])
    D_B.model.load_state_dict(save_info['D_B_state'])
if not os.path.exists(img_path):
    os.makedirs(img_path)
print("cuda status : ,device", is_cuda, device)
if is_cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()

# for model in D_A.model:
#     print(model.device)
# print(D_A.model.device)
# loss define
loss_GAN = torch.nn.MSELoss()
loss_cycle = torch.nn.L1Loss()
loss_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                               lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(epochs, now_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(epochs, now_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(epochs, now_epoch, decay_epoch).step)

# Inputs & targets memory allocation
target_real = torch.ones(size=(batch_size, 1), requires_grad=False)
target_fake = torch.zeros(size=(batch_size, 1), requires_grad=False)
if is_cuda:
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()
# print(target_real.is_cuda)
trans = [
    transforms.Resize(int(256 * 1.12), Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
horse_zebra_dataset = ImageDataSet(dataset='horse2zebra', zip_path=z_path, transforms_=trans)
train_loader = torch.utils.data.DataLoader(horse_zebra_dataset, batch_size=batch_size, shuffle=True)

for epo in range(now_epoch, epochs):
    time_start = time.time()
    print("train epoch {} of {} ".format(epo, epochs))
    cnt = 0
    running_G_loss = 0
    running_D_A_loss = 0
    running_D_B_loss = 0
    for i, (t_A, t_B) in enumerate(train_loader):
        cnt += 1
        if is_cuda:
            t_A = t_A.cuda()
            t_B = t_B.cuda()
        #          train gen
        #         print(t_A.shape)
        optimizer_G.zero_grad()
        same_B = G_AB(t_B)
        same_A = G_AB(t_A)

        loss_identity_B = loss_identity(same_B, t_B) * 5
        loss_identity_A = loss_identity(same_A, t_A) * 5

        fake_B = G_AB(t_A)
        fake_A = G_BA(t_B)
        pred_fake_B = D_B(fake_B)
        pred_fake_A = D_A(fake_A)

        loss_GAN_AB = loss_GAN(pred_fake_B, target_real)
        loss_GAN_BA = loss_GAN(pred_fake_A, target_real)

        # cycyle loss
        recovered_A = G_BA(fake_B)
        recovered_B = G_AB(fake_A)
        loss_cycle_A = loss_cycle(recovered_A, t_A) * 10.0
        loss_cycle_B = loss_cycle(recovered_B, t_B) * 10.0

        loss_G = loss_identity_A + loss_GAN_AB + loss_cycle_A + loss_cycle_B + loss_GAN_BA + loss_identity_B
        loss_G.backward()
        optimizer_G.step()

        # D_A train
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = D_A(t_A)
        loss_D_real = loss_GAN(pred_real, target_real)

        # Fake loss
        # fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_AA = G_BA(t_B)
        pred_fake = D_A(fake_AA.detach())
        loss_D_fake = loss_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = D_B(t_B)
        loss_D_real = loss_GAN(pred_real, target_real)

        # Fake loss
        fake_BB = G_AB(t_A)
        pred_fake = D_B(fake_BB.detach())
        loss_D_fake = loss_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        running_G_loss += loss_G
        running_D_A_loss += loss_D_A
        running_D_B_loss += loss_D_B

        ###################################
        # print("loss G:{},loss_D_A:{},loss_D_B:{}".format(loss_G.detach().item(), loss_D_A.detach().item(),
        #                                                  loss_D_B.detach().item()))
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    running_G_loss /= cnt
    running_D_A_loss /= cnt
    running_D_B_loss /= cnt
    print("time spend {}".format(time.time() - time_start))
    if epo % 2 == 0:
        torch.save(
            {"epoch": epo, 'g_AB_state': G_AB.model.state_dict(), 'g_BA_state': G_BA.model.state_dict(),
             'd_A_state': D_A.model.state_dict(), 'd_B_state': D_B.state_dict()
             }, "cyclegan.pkl")
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write("epoch:{},lr:{} G_loss:{},D_A_loss:{},D_B_loss:{}"
                .format(epo, optimizer_G.state_dict()['param_groups'][0]['lr'], running_G_loss, running_D_A_loss,
                        running_D_B_loss)
                )
