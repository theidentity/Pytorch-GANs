import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset


class Data_IO():

    def __init__(self,lab_per_class,batch_size):

        self.lab_per_class = lab_per_class
        self.img_sz = 32
        self.img_ch = 1
        self.batch_size = batch_size

    def get_original_dataset(self,split):
        transform = transforms.Compose([
            transforms.Resize(self.img_sz),
            transforms.CenterCrop(self.img_sz),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # dataset = datasets.SVHN(
            # 'data/svhn/', split=split, download=True, transform=transform)
        train = split in ('train')
        dataset = datasets.CIFAR10(
            'data/cifar10/', train=train, download=True, transform=transform)
        return dataset


    def get_dataset(self,split):
        if split == 'all_train':
            dataset = self.get_original_dataset(split='train')
            return dataset

        elif split == 'test':
            dataset = self.get_original_dataset(split='test')
            return dataset

        elif split == 'lab_train':
            dataset = self.get_original_dataset(split='train')
            y = torch.Tensor(dataset.labels)
            idx = torch.arange(len(y))
            cl_items = torch.unique(y)
            req_idx = torch.empty((len(cl_items)*self.lab_per_class,))

            for i,cl in enumerate(cl_items):
                req_idx[i*self.lab_per_class:(i+1)*self.lab_per_class] = idx[y==cl][:self.lab_per_class]

            req_idx = req_idx.type(dtype=torch.long)
            train_lab = Subset(dataset,req_idx)
            return train_lab

    def get_dataloader(self,split):
        assert split in ('all_train','test','lab_train')
        dataset = self.get_dataset(split)
        shuffle = False if split == 'test' else True
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=16)
        return dataloader


class Infinite_Dataloader():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.len = len(dataloader)

    def __len__(self):
        return self.len

    def __iter__(self):
        while True:
            for item in iter(self.dataloader):
                yield item

if __name__ == '__main__':
    io = Data_IO(lab_per_class=50,batch_size=32)
    dataloader = io.get_dataloader(split='all_train')
    # dataloader = io.get_dataloader(split='test')
    # dataloader = io.get_dataloader(split='lab_train')

    nr_batches = len(dataloader)
    dataloader = Infinite_Dataloader(dataloader)
    dataloader = iter(dataloader)

    for i in range(nr_batches):
        x,y = next(dataloader)
        print(x.shape,y)
