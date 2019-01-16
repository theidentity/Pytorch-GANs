import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
	def forward(self,x):
		return x.view(x.shape[0],-1)


class Reshape(nn.Module):
	def __init__(self, target_shape):
		super(Reshape, self).__init__()
		self.target_shape = (-1,) + target_shape

	def forward(self,x):
		return x.view(self.target_shape)


class GlobalAveragePooling2D(nn.Module):
    def forward(self,x):
        x = F.adaptive_avg_pool2d(x,(1,1))
        return x    