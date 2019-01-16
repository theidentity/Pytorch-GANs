import torch.nn as nn
import torch.nn.functional as F
import pt_layers as ptl
from torch.nn.utils import weight_norm


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_dim = 100
        self.layers = nn.Sequential(

            nn.Linear(self.latent_dim,512*4*4),
            ptl.Reshape((512,4,4)),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256, momentum=0.1),
            nn.LeakyReLU(.2, inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128, momentum=0.1),
            nn.LeakyReLU(.2, inplace=True),

            weight_norm(nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
            )

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(

            nn.Dropout2d(p=.2),
            weight_norm(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(p=.5),
            weight_norm(nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout2d(p=.5),
            weight_norm(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)),
        
            ptl.GlobalAveragePooling2D(),
            ptl.Flatten(),
            )

        self.pred_layers = nn.Sequential(
            weight_norm(nn.Linear(192,1)),
            nn.Sigmoid(),
            )


    def forward(self, img, req_inter=False):
        inter_layer = self.conv_layers(img)
        pred = self.pred_layers(inter_layer)
        if req_inter:
            return inter_layer,pred
        return pred
        

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    # elif classname.find('Linear') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)



if __name__ == '__main__':
    import torch
    batch_size = 128
    z = torch.randn((batch_size, 100))

    G = Generator().apply(weights_init)
    print(G(z).shape)

    D = Discriminator().apply(weights_init)
    features,logits = D(G(z))
    print(features.shape)
    print(logits.shape)