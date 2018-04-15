#importing all the required libraries

%matplotlib inline
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import *
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as autograd
from torch.optim.lr_scheduler import MultiStepLR
from blur_face import blur_face
import cv2
import tflib as lib
from plot import *
import os
import numpy as np
from glob import glob
from PIL import *
from tqdm import tqdm
import torchvision.datasets as dset
from torch.optim import lr_scheduler

import time
import os
import os.path

#Define functions

def default_loader(path):
    return Image.open(path[0])

def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            item = (path, 0)
            images.append(item)

    return images

#making the class to load Dataset

class LoadDataset(data.Dataset):

    def __init__(self, root, transform=transforms.Compose([transforms.ToTensor()]),
                 loader=default_loader):
        print("Root :",root+"/origin/")
        y_imgs = make_dataset(root+"/origin/")
#         x_imgs = make_dataset(root+"/small/")
        if  (len(y_imgs) == 0):
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        print("Found {} images in subfolders of: {}".format(len(y_imgs), root))

        self.root = root
#         self.x_imgs = x_imgs
        self.y_imgs = y_imgs
        self.transform = transform
#         self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        target_path = self.y_imgs[index]
#         path = self.x_imgs[index]
#         img = self.loader(path)
        target = self.loader(self.y_imgs[index])
#         if self.transform is not None:
#             img = self.transform(img)
        if self.transform is not None:
            target = self.transform(target)

        return target

    def __len__(self):
        return len(self.y_imgs)

#Data loader

def get_loader(root, batch_size, scale_size, num_workers=4, shuffle=True):
    dataset_name = os.path.basename(root)
    image_root = root

    dataset = LoadDataset(root=image_root, transform=transforms.Compose([
        transforms.Resize(scale_size),
        transforms.ToTensor(),
    ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader


DIM = 128              # This overfits substantially; you're probably better off with 64
LAMBDA = 10            # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5       # How many critic iterations per generator iteration
BATCH_SIZE = 1        # Batch size
ITERS = 200000         # How many generator iterations to train for
OUTPUT_DIM = 3*128*128 # Number of pixels in our origin (3*128*128)

#

trainloader = get_loader('data/celeba', BATCH_SIZE, DIM)

# testloader = get_loader('data/celeba/test',BATCH_SIZE, DIM)

#Generative Adverserial Network Implimentation

#Generator Class

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Conv2d(3,  8 * DIM, 3, 2,padding = 1),
            nn.BatchNorm2d( 8 * DIM),
            nn.LeakyReLU(),
            nn.Conv2d(8*DIM, 4*DIM, 3, 2,padding = 1),
            nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU(),
            nn.Conv2d(4*DIM, int(DIM*2), 3, 2,padding = 1),
            nn.BatchNorm2d(int(DIM*2)),
           nn.LeakyReLU(),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(int(DIM*2),  int(DIM), 2, stride=2),
            nn.BatchNorm2d(int(DIM)),
            nn.LeakyReLU(),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d( int(DIM), int(DIM/2), 2, stride=2),
            nn.BatchNorm2d(int(DIM/2)),
            nn.LeakyReLU(),

            nn.ConvTranspose2d( int(DIM/2), int(DIM/4), 2, stride=2),
            nn.BatchNorm2d(int(DIM/4)),
             nn.LeakyReLU(),
        )
        block3= nn.Sequential(
            nn.ConvTranspose2d( int(DIM/4), int(DIM/8), 2, stride=1),
            nn.BatchNorm2d(int(DIM/8)),
            nn.LeakyReLU(),

            nn.ConvTranspose2d( int(DIM/8), 3, 2, stride=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(3, 3, 3, stride=1,padding =2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),

        )
        deconv_out = nn.ConvTranspose2d(3, 3, 3, stride=1,padding=1)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
#         print(input.size())
        output = self.preprocess(input)
#         output = output.view(-1, 4 * DIM, 4, 4)
#         print(output.size())
        output = self.block1(output)
#         print(output.size())
        output = self.block2(output)
#         print(output.size())
        output = self.block3(output)
        print(output.size())
        output = self.deconv_out(output, output_size = input.size())
        print(output.size())
        output = self.tanh(output)
        return output.view(-1, 3, 128, 128)

#Discriminator class

    class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 3, stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 3, 3, stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(3, int(DIM/8), 3, stride = 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(DIM/8), int(DIM/4), 3, stride = 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(DIM/4), int(DIM/2), 3, stride = 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(DIM/2), int(DIM), 3, stride = 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(DIM), int(DIM*2), 3, stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * DIM, 8 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

#Instantiating
netG = Generator()
netD = Discriminator()
print (netG)
print (netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)


one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)
print(one,mone)

#This part makes this gan a wgan-gp

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

schedD = lr_scheduler.CosineAnnealingLR(optimizerD, 4000, eta_min=1e-5, last_epoch=-1)
schedG = lr_scheduler.CosineAnnealingLR(optimizerG, 4000, eta_min=1e-5, last_epoch=-1)


def calc_gradient_penalty(netD, real_data, fake_data):
#     print( "real_data: ", real_data.size(), fake_data.size(),BATCH_SIZE, real_data.nelement()/BATCH_SIZE)
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 128, 128)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


#For Generating Samples

    def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 128, 128)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, './tmp/celebA/samples_{}.jpg'.format(frame))

# For calculating inception score

def get_inception_score(G, ):
    all_samples = []
    for i in xrange(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator

# train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images in trainloader:
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)

            yield images
# def inf_test_gen():
#     while True:
#         for images in testloader:
#             # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)

#             yield images
gen = inf_train_gen()
# dev_gen = inf_test_gen()
preprocess = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

# Train Loop

scale = transforms.Compose([
                            transforms.Resize(32),
                            transforms.Resize(128),
                            transforms.ToTensor(),
#                             transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                                 std = [0.229, 0.224, 0.225])

                            ])

convert_pil = transforms.Compose([transforms.ToPILImage(),

#                             transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                                 std = [0.229, 0.224, 0.225])

                            ])


my_file = Path("./netD.pth")
if my_file.is_file():
    netD.load_state_dict(torch.load('netD.pth', map_location=lambda storage, loc: storage))
my_file1 = Path("./netG.pth")
if my_file1.is_file():
    netG.load_state_dict(torch.load('netG.pth', map_location=lambda storage, loc: storage))


for iteration in tqdm(range(ITERS)):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        _data = next(gen)
        noise = torch.FloatTensor(BATCH_SIZE, 3, 128, 128)
        netD.zero_grad()
        # train with real
        _data = _data.view(BATCH_SIZE, 3, 128, 128)
        #_data = _data.permute(0, 2, 3, 1)
        for j in range(BATCH_SIZE):
            pilimg = convert_pil(_data[j])
            im1 = pilimg.filter(ImageFilter.BLUR)
            noise[j] = scale(im1)

        real_data = torch.stack([item for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = Variable(real_data)

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)

        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        noise = torch.stack([item for item in noise])
        if use_cuda:
            noise = noise.cuda(gpu)

        noisev = Variable(noise, volatile=True)  # totally freeze netG
        fake = Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()
    _data = next(gen).view(BATCH_SIZE, 3, 128, 128)
    noise = torch.FloatTensor(BATCH_SIZE, 3, 128, 128)
    for j in range(BATCH_SIZE):
            noise[j] = scale(_data[j])
    if use_cuda:
        noise = noise.cuda(gpu)
#     noise = torch.stack([item for item in noise])
#     print(noise.size())
    noisev = Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    plot('./tmp/celebA/train disc cost', D_cost.cpu().data.numpy())
    plot('./tmp/celebA/time', time.time() - start_time)
    plot('./tmp/celebA/train gen cost', G_cost.cpu().data.numpy())
    plot('./tmp/celebA/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate inception score every 1K iters
    if False and iteration % 1000 == 999:
        inception_score = get_inception_score(netG)
        plot('./tmp/celebA/inception score', inception_score[0])

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 100 == 99:
        dev_disc_costs = []
        i=0
        for images in (dev_gen):
            print (str(i)+"\r",end="")
            i+=1
            images = images.view(BATCH_SIZE, 3, 128, 128)
#             images = images.permute(0, 2, 3, 1)
            imgs = torch.stack([item for item in images])

            # imgs = preprocess(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = Variable(imgs, volatile=True)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        plot('./tmp/celebA/dev disc cost', np.mean(dev_disc_costs))

#         generate_image(iteration, netG)

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        flush()
    tick()
    if(iteration % 50 == 0):
        print("Saved !",end="\r")
        torch.save(netD.state_dict(), './netD.pth')
        torch.save(netG.state_dict(), './netG.pth')

        vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ('./results',iteration), normalize= True)


from PIL import Image

rescale = transforms.Compose([
                            transforms.Resize(128),
                            transforms.ToTensor(),
#                             transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                                 std = [0.229, 0.224, 0.225])

                            ])
rescale_1 = transforms.Compose([
                            transforms.Resize(128),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                                 std = [0.229, 0.224, 0.225])

                            ])

im = Image.open("input.jpg")
a = rescale(im)
a = Variable(a)
a = netG(a.view(-1,3,128,128))
# rescale_1(a)
vutils.save_image(a.data, 'result.png', normalize= True)
