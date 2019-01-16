import os
import shutil
import torch.nn as nn


def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

class Flatten(nn.Module):
	"""docstring for Flatten"""
	def __init__(self, arg):
		super(Flatten, self).__init__()
		self.arg = arg
		