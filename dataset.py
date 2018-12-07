from PIL import Image
import torch
import os
from torch.utils.data import Dataset

defect_class = {'Spot':0,'Mold':1,'Scratch':2,'Hole':3,'Caterpillar':4,'Flat flower':5,
                'Warped':6,'Zinc ash':7,'Dirty':8,'Zinc residue':9,'pressing':9}

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        imgs = []
        for dir in os.listdir(root):
            path = os.path.join(root, dir)
            label = defect_class[dir]
            for file in os.listdir(path):
                img = os.path.join(path,file)
                imgs.append([img, label])
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    trainData = MyDataset('augLib/test')
    print('')