import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn

import numpy as np
from tqdm import tqdm
import argparse
import os

import helpers
from data_io import Data_IO,Infinite_Dataloader
from networks import Generator,Discriminator,weights_init


class GAN(object):
    def __init__(self):

        self.img_sz = 32
        self.img_ch = 3
        self.batch_size = 128
        self.latent_dims = 100

        self.name = 'dcgan'
        self.save_path = 'models/'+self.name+'/'
        self.gen_img_dir = 'gen_imgs/'+self.name+'/'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.lab_per_class = 20
        self.io = Data_IO(lab_per_class=self.lab_per_class,batch_size=self.batch_size)

    def get_data(self,split):
        assert split in ('all_train','test','lab_train')
        dataloader = self.io.get_dataloader(split=split)
        dataloader = Infinite_Dataloader(dataloader)
        nr_batches = len(dataloader)
        infinite_datagenerator = iter(dataloader)
        return nr_batches, infinite_datagenerator

    def get_models(self):
        G = Generator().to(self.device)
        D = Discriminator().to(self.device)
        G.apply(weights_init)
        D.apply(weights_init)
        return G, D


    def train(self, num_epochs=1):

        G, D = self.get_models()
        criterion = torch.nn.BCELoss()
        fixed_noise = torch.randn(self.batch_size, self.latent_dims).to(self.device)
        G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(.5, .999))
        D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(.5, .999))
        nr_train_batches,train_dataloader = self.get_data('all_train')

        helpers.clear_folder(self.gen_img_dir)
        helpers.clear_folder(self.save_path)

        for epoch in range(num_epochs):
            for i in tqdm(range(nr_train_batches)):
                real_imgs,__ = next(train_dataloader)

                real = torch.full((real_imgs.shape[0], 1), 1).to(self.device)
                fake = torch.full((real_imgs.shape[0], 1), 0).to(self.device)
                real_imgs = real_imgs.to(self.device)

                D.zero_grad()
                D_real_loss = criterion(D(real_imgs), real)
                D_real_loss.backward()

                z = torch.randn((real_imgs.shape[0], 100)).to(self.device)
                gen_imgs = G(z).detach()
                D_fake_loss = criterion(D(gen_imgs), fake)
                D_fake_loss.backward()

                D_loss = (D_real_loss + D_fake_loss)/2.0
                D_opt.step()

                G.zero_grad()
                z = torch.randn((real_imgs.shape[0], 100)).to(self.device)
                gen_imgs = G(z)
                G_loss = criterion(D(gen_imgs), real)
                G_loss.backward()
                G_opt.step()

                if i % 100 == 0:
                    print('Epoch %d Step %d D_loss %.4f G_loss %.4f ' %(epoch, i, D_loss.item(), G_loss.item()))
                    name = str(epoch).zfill(4)+str(i).zfill(4) + '.png'
                    save_image(G(fixed_noise)[:25], self.gen_img_dir+name, nrow=5, normalize=True)

            name = str(epoch).zfill(4)
            torch.save(G.state_dict(), self.save_path+name+'_gen.pth')
            torch.save(D.state_dict(), self.save_path+name+'_disc.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', default='42')
    parser.add_argument('--epochs', default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_epochs = int(args.epochs)
    
    gan = GAN()
    gan.train(num_epochs=num_epochs)