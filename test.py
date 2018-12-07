import os
import numpy as np
import torch
from torchvision import transforms
from dataset import MyDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

test_path = 'data/augLib-11-100_5/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model/TSteel_inception_10_kinds.pth').to(device)

transform = transforms.Compose([
    transforms.Resize(299),  # 将图像转化为32 * 32
    # transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
    # transforms.RandomCrop(24),  # 从图像中裁剪一个24 * 24的
    # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

test_datasets = MyDataset(test_path, transform = transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=22,
                                          shuffle=True)
def class_acc(label, predict):
    length = len(label)
    acc = np.zeros((11,11))
    acc_rate = [0 for i in range(11)]
    for i in range(length):
        acc[label[i]][predict[i]] += 1
        if label[i] == predict[i]:
            acc_rate[label[i]] += 1
    for i in range(11):
        acc_rate[i] /= sum(acc[0])

    print(acc)
    print(acc_rate)

total, correct_prediction = 0, 0
label, predict = [], []
for images, labels in test_loader:
    # to GPU
    images = images.to(device)
    labels = labels.to(device)
    # print prediction
    outputs = model(images)
    # equal prediction and acc
    _, predicted = torch.max(outputs.data, 1)

    for i in labels:
        label.append(i)
    for j in predicted:
        predict.append(j)

    # val_loader total3acf
    total += labels.size(0)
    # add correct
    correct_prediction += (predicted == labels).sum().item()

class_acc(label, predict)
print(f"test_acc: {(correct_prediction / total):4f}")

