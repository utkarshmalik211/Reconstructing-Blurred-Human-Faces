from __future__ import print_function
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from plot import *
from PIL import *
from PIL import Image



DIM = 128


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Conv2d(3, 8 * DIM, 3, 2, padding=1),
            nn.BatchNorm2d(8 * DIM),
            nn.LeakyReLU(),
            nn.Conv2d(8 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU(),
            nn.Conv2d(4*DIM, int(DIM*2), 3, 2, padding=1),
            nn.BatchNorm2d(int(DIM*2)),
            nn.LeakyReLU(),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(int(DIM*2), int(DIM), 2, stride=2),
            nn.BatchNorm2d(int(DIM)),
            nn.LeakyReLU(),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(int(DIM), int(DIM/2), 2, stride=2),
            nn.BatchNorm2d(int(DIM/2)),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(int(DIM/2), int(DIM/4), 2, stride=2),
            nn.BatchNorm2d(int(DIM/4)),
            nn.LeakyReLU(),
        )
        block3 = nn.Sequential(
            nn.ConvTranspose2d(int(DIM/4), int(DIM/8), 2, stride=1),
            nn.BatchNorm2d(int(DIM/8)),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(int(DIM/8), 3, 2, stride=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(3, 3, 3, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),

        )
        deconv_out = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        print(output.size())
        output = self.deconv_out(output, output_size=input.size())
        print(output.size())
        output = self.tanh(output)
        return output.view(-1, 3, 128, 128)


netG = Generator()
print(netG)


use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    netG = netG.cuda(gpu)


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

convert_pil = transforms.Compose([transforms.ToPILImage(),])

my_file1 = Path("./netG.pth")
if my_file1.is_file():
    netG.load_state_dict(torch.load('netG.pth', map_location=lambda storage, loc:storage))

rescale = transforms.Compose([
                            transforms.Resize(128),
                            transforms.ToTensor(),
#                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                                 std = [0.229, 0.224, 0.225])

                            ])

im = Image.open("input.jpg")
a = rescale(im)
a = Variable(a)
a = netG(a.view(-1, 3, 128, 128))
# rescale_1(a)
vutils.save_image(a.data, 'result.png', normalize=True)
