import os, sys
sys.path.append(os.getcwd())

import time

from utils.utility import mkdir_p, generate_image
from utils.plot import plot, flush

import numpy as np


import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch import autograd
from torch import optim
import argparse
import csv




parser = argparse.ArgumentParser(description='parse the input options')

parser.add_argument('--name', type=str, default='cifar10', help='name of the experiment. It decides where to store the results and checkpoints')
parser.add_argument('--results_dir', type=str, default='./results', help='folder to store the results')
parser.add_argument('--image_size', type=int, default=32, help='input image size, for cifar10 is 32x32')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--workers', type=int, default=2, help='# of workers to load the dataset')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='folder to store the model checkpoint')
parser.add_argument('--noise_dim', type=int, default=100, help='input dim of noise')
parser.add_argument('--dim', type=int, default=64, help='# of filters in first conv layer of both discrim and gen')
parser.add_argument('--data_dir', required=True, help='folder of the dataset')
parser.add_argument('--netG', type=str, default='', help='checkpoints of netG you wish to use in continuing the training')
parser.add_argument('--netD', type=str, default='', help='checkpoints of netD you wish to use in continuing the training')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--num_epochs', type=int, default=200, help='# of epochs to train')




opt = parser.parse_args()



dtype = torch.FloatTensor

mkdir_p(os.path.join(opt.results_dir,opt.name))
mkdir_p(os.path.join(opt.checkpoints_dir,opt.name))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( opt.noise_dim, opt.dim * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(opt.dim * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (opt.dim*8) x 4 x 4
                nn.ConvTranspose2d(opt.dim * 4, opt.dim * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.dim * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (opt.dim*4) x 8 x 8
                nn.ConvTranspose2d(opt.dim * 2, opt.dim, 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.dim),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (opt.dim*2) x 16 x 16
                nn.ConvTranspose2d(opt.dim, 3, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 32 x 32
            )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),#64x16x16
            nn.Conv2d(opt.dim, 2 * opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),#128x8x8
            nn.Conv2d(2 * opt.dim, 4 * opt.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),#256x4x4
            nn.Conv2d(4*opt.dim, 4*opt.dim, 4),
            nn.BatchNorm2d(4*opt.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),#256x1x1
            nn.Conv2d(4*opt.dim, 10, 1)
        )

        self.main = main
        self.softmax = nn.Softmax()

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 10)
        output = self.softmax(output)
        return output







#marginalized entropy
def entropy1(y):
    y1 = autograd.Variable(torch.randn(y.size(1)).type(dtype), requires_grad=True)
    y2 = autograd.Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = y.mean(0)
    y2 = -torch.sum(y1*torch.log(y1+1e-6))

    return y2



# entropy
def entropy2(y):
    y1 = autograd.Variable(torch.randn(y.size()).type(dtype), requires_grad=True)
    y2 = autograd.Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = -y*torch.log(y+1e-6)

    y2 = 1.0/opt.batch_size*y1.sum()
    return y2








netG = Generator()
netD = Discriminator()

#continue traning by loading the latest model or the model specified in --netG and --netD
if opt.continue_train:
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    else:
        netG.load_state_dict(torch.load('%s/netG_latest.pth' % (os.path.join(opt.checkpoints_dir,opt.name))))


    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    else:
        netD.load_state_dict(torch.load('%s/netD_latest.pth' % (os.path.join(opt.checkpoints_dir,opt.name))))



print netG
print netD

use_cuda = torch.cuda.is_available()

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

one = torch.FloatTensor([1])
mone = one * -1

if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9))




# Dataset iterator
dataset = dset.ImageFolder(root=opt.data_dir,
                           transform=transforms.Compose([
                               transforms.Scale(opt.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

print("Start training on %s dataset which contains %d images..." %(opt.name, len(dataset)))






iter_idx = 0
with open(os.path.join(opt.results_dir, opt.name, 'log.csv'), 'wb') as log:
    log_writer = csv.writer(log, delimiter=',')

    for epoch in xrange(opt.num_epochs):
        start_time = time.time()

        for batch_idx, (real, labels) in enumerate(dataloader):
            ###########################
            # (1) Update D network
            ###########################

            #freeze G and update D
            for p in netD.parameters():  
                p.requires_grad = True  
            for p in netG.parameters(): 
                p.requires_grad = False  
            netD.zero_grad()

            #################
            # train with real
            #################
            if use_cuda:
                real = autograd.Variable(real.cuda())

            D_real = netD(real)
            # minimize entropy to make certain prediction of real sample
            entorpy2_real = entropy2(D_real)
            entorpy2_real.backward(one, retain_graph=True)

            # maximize marginalized entropy over real samples to ensure equal usage
            entropy1_real = entropy1(D_real)
            entropy1_real.backward(mone)

            #################
            # train with fake
            #################
            noise = torch.randn(opt.batch_size, opt.noise_dim, 1, 1)
            if use_cuda:
                noise = autograd.Variable(noise.cuda())  # totally freeze netG

            fake = netG(noise)
            D_fake = netD(fake)

            #minimize entropy to make uncertain prediction of fake sample
            entorpy2_fake = entropy2(D_fake)
            entorpy2_fake.backward(mone)


            D_cost = entropy1_real + entorpy2_real + entorpy2_fake
            optimizerD.step()
            ############################
            # (2) Update G network
            ###########################

            #freeze D and update G
            for p in netD.parameters():
                p.requires_grad = False  
            for p in netG.parameters():
                p.requires_grad = True  
            netG.zero_grad()


            noise = torch.randn(opt.batch_size, opt.noise_dim, 1, 1)
            noise = autograd.Variable(noise.cuda())
            fake = netG(noise)
            D_fake = netD(fake)

            #fool D to make it believe the generated samples are real
            entropy2_fake = entropy2(D_fake)
            entropy2_fake.backward(one, retain_graph=True)

            #ensure equal usage of fake samples
            entropy1_fake = entropy1(D_fake)
            entropy1_fake.backward(mone)

            G_cost = entropy2_fake + entropy1_fake
            optimizerG.step()


            D_cost = D_cost.cpu().data.numpy()
            G_cost = G_cost.cpu().data.numpy()
            entorpy2_real = entorpy2_real.cpu().data.numpy()
            entorpy2_fake = entorpy2_fake.cpu().data.numpy()

            #monitoring the loss
            plot('errD', D_cost, iter_idx)
            # plot('time', time.time() - start_time, iter_idx)
            plot('errG', G_cost, iter_idx)
            plot('errD_real', entorpy2_real, iter_idx)
            plot('errD_fake', entorpy2_fake, iter_idx)


            # Save plot every  iter
            flush(os.path.join(opt.results_dir, opt.name))

            # Write losses to logs 
            log_writer.writerow([D_cost[0],G_cost[0],entorpy2_real[0],entorpy2_fake[0]])

            print "iter%d[epoch %d]\t %s %.4f \t %s %.4f \t %s %.4f \t %s %.4f" % (iter_idx, epoch,
                                                         'errD', D_cost,
                                                         'errG', G_cost,
                                                         'errD_real', entorpy2_real,
                                                         'errD_fake', entorpy2_fake )

            #checkpointing the latest model every 500 iteration
            if iter_idx % 500 == 0:
                torch.save(netG.state_dict(), '%s/netG_latest.pth' % (os.path.join(opt.checkpoints_dir,opt.name)))
                torch.save(netD.state_dict(), '%s/netD_latest.pth' % (os.path.join(opt.checkpoints_dir,opt.name)))

            iter_idx += 1


        # generate samples every 2 epochs for surveillance
        if epoch % 2 == 0:
            generate_image(epoch, netG, opt)


        # do checkpointing every 20 epochs
        if epoch % 20 == 0:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(opt.checkpoints_dir, opt.name), epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(opt.checkpoints_dir, opt.name), epoch))
