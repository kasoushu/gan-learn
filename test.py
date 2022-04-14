import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import os
# 超参数
batch_size = 128
num_epoch = 500
z_dimention = 100
d_optimizer_lr = 0.0001
g_optimizer_lr = 0.0001
use_gpu = True


class Discriminator(nn.Module):
    # 784长度向量 -> (0,1)
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x
class generator(nn.Module):  # 100长度向量 -> 784长度向量
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x

# 784向量 -> 1*28*28
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=True
)

dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size,shuffle=True,drop_last=True
)

if __name__=='__main__':
    if not os.path.exists('./img'):
        os.mkdir('./img')
    D = Discriminator()
    G = generator()
    criterion = nn.BCELoss()  # 二分类的交叉熵
    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_optimizer_lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_optimizer_lr)
    z=torch.FloatTensor(batch_size,z_dimention)
    if use_gpu:
        D = D.cuda()
        G = G.cuda()
        z = z.cuda()
    for epoch in range(num_epoch):
        print("epoch ",epoch, "  of  ",num_epoch)
        for i, (img, _) in enumerate(dataloader):
            img = img.view(batch_size, -1)
            if use_gpu:
                real_img = img.cuda()
                real_label = torch.ones(batch_size).cuda()
                fake_label = torch.zeros(batch_size).cuda()
            else:
                real_img = img
                real_label = torch.ones(batch_size).view(batch_size,1)
                fake_label = torch.zeros(batch_size).view(batch_size,1)
            # =================train discriminator
            real_out = D(real_img)
            d_loss_real = criterion(real_out, real_label)  # 真实数据对应输出 1
            real_scores = real_out

            z.data.normal_(0,1)
            fake_img1 = G(z)
            fake_out = D(fake_img1.detach())
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===============train generator
            z.data.normal_(0,1)
            fake_img2 = G(z)
            output = D(fake_img2)
            g_loss = criterion(output, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')
        print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} D real:{:.6f},D fake:{:.6f}'.format(
            epoch, num_epoch, d_loss.item(), g_loss.item(), real_scores.data.mean(), fake_scores.data.mean()
        ))
        fake_images = to_img(fake_img2.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
