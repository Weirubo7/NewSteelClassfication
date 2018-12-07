import argparse
import os
import time
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from dataset import MyDataset
from SENet import se_resnet18

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
# parser.add_argument('--path', type=str, default='data/cifar10/',
#                     help="""image dir path default: 'data/cifar10/'.""")

parser.add_argument('--epochs', type=int, default=100,
                    help="""Epoch default:50.""")
parser.add_argument('--train_batch_size', type=int, default=32,
                    help="""Batch_size default:256.""")
parser.add_argument('--test_batch_size', type=int, default=22,
                    help="""Batch_size default:110.""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='model/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='TSteel_inception_10_kind.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=1)

args = parser.parse_args()

train_path = 'data/augLib-11-100_2/train'
test_path = 'data/augLib-11-100_2/test'
log_txt = 'logs/TSteel_alex_p2.txt'

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(299),  # 将图像转化为32 * 32
    # transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
    # transforms.RandomCrop(24),  # 从图像中裁剪一个24 * 24的
    # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

train_datasets = MyDataset(train_path, transform = transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=args.train_batch_size,
                                           shuffle=True)

test_datasets = MyDataset(test_path, transform = transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=args.test_batch_size,
                                          shuffle=True)

# 动态调整学习率
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr * (0.95 ** (epoch ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    print(f"Train numbers:{len(train_datasets)}")

    # Alexnet
    # model = torchvision.models.alexnet(pretrained=True).to(device)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    # model.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(4096, args.num_classes),
    #     ).to(device)
    # model.classifier[6] = nn.Linear(4096, args.num_classes).to(device)
    # print(model)

    # Vgg16
    # model = torchvision.models.vgg16_bn(pretrained=True).to(device)
    # model.classifier[6] = nn.Linear(4096,args.num_classes).to(device)
    # print(model)

    # Inception_v3
    # model = torchvision.models.inception_v3(pretrained=True).to(device)
    # model.aux_logits = False
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features,args.num_classes).to(device)
    # print(model)

    model = torchvision.models.resnet50(pretrained=False)
    print(model)

    # resnet18 + SENet
    # model = se_resnet18(num_classes=args.num_classes).to(device)
    # print(model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    # if os.path.exists(log_txt):
    #     os.remove(log_txt)
    # f = open(log_txt, 'a')

    test_acc_list = [0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        # start time
        start = time.time()
        correct_prediction = 0.
        total = 0
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            # val_loader total3acf
            total += labels.size(0)
            # add correct
            correct_prediction += (predicted == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = correct_prediction / total

        adjust_learning_rate(optimizer, epoch)

        if epoch % args.display_epoch == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Train_acc {train_acc: .4f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            model.eval()

            correct_prediction = 0.
            total = 0
            for images, labels in test_loader:
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                # val_loader total3acf
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            test_acc = round((correct_prediction / total),4)

            # 模型效果好于之前的时候才会保存模型
            if test_acc >= max(test_acc_list):
                torch.save(model, args.model_path + args.model_name)

            test_acc_list.append(test_acc)
            print(f"test_acc: {test_acc}, "
                  f"max_test_acc: {max(test_acc_list)}")

            string = 'epoch:' + str(epoch) + '\t' + 'accuracy:' + str(test_acc) + '\n'
            # f.write(string)

    # f.close()

    # Save the model checkpoint
    # torch.save(model, args.model_path + args.model_name)
    # print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    train()
