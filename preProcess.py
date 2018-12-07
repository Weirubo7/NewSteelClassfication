import os
import random
dict = {
    'Spot':'斑迹',
    'Mold':'保护渣',
    'Scratch':'划伤',
    'Hole':'孔洞',
    'Caterpillar':'毛毛虫',
    'Flat flower':'平整花',
    'Warped':'翘皮',
    'Zinc ash':'锌灰',
    'Zinc residue':'锌渣',
    'Dirty':'脏污',
    'pressing':'异物压入'
}

import cv2
def preProcess(path):
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]

    img_roi = img.copy()
    if width > height:
        x_left = int((width - height) / 2)
        x_right = x_left + height
        img_roi = img[0:height,x_left:x_right]
    elif width < height:
        y_top = int((height - width) / 2)
        y_down = y_top + width
        img_roi = img[y_top:y_down,0:width]

    # cv2.imshow('tsy',img_roi)
    # cv2.waitKey(1000)

    return img_roi

def do_process():
    oldDir = 'defect2/lib-11-100'
    newDir = 'defect2/proLib-11-100'
    if not os.path.exists(newDir):
        os.mkdir(newDir)
    for dir in os.listdir(oldDir):
        defectPath = os.path.join(oldDir, dir)
        if os.path.isdir(defectPath):
            newPath = os.path.join(newDir, dir)
            if not os.path.exists(newPath):
                os.mkdir(newPath)
            for file in os.listdir(defectPath):
                imgPath = os.path.join(defectPath, file)
                newImagePath = os.path.join(newPath, file)
                img_roi = preProcess(imgPath)
                cv2.imwrite(newImagePath, img_roi)

def do_dataAugment():
    oldDir = 'data/proLib-11-100'
    newDir = 'data/augLib-11-100_4'
    for dir in os.listdir(oldDir):
        defectPath = os.path.join(oldDir, dir)
        if os.path.isdir(defectPath):
            trainPath = os.path.join(newDir, 'train/'+ dir)
            testPath = os.path.join(newDir, 'test/'+ dir)
            if not os.path.exists(trainPath):
                os.makedirs(trainPath)
            if not os.path.exists(testPath):
                os.makedirs(testPath)
            perKindList = []  # 每类中所有的图片
            for file in os.listdir(defectPath):
                perKindList.append(file)
            random.shuffle(perKindList)
            train_list = perKindList[0:-10] # 训练集
            test_list = perKindList[-10:-1]
            test_list.append(perKindList[-1]) # 测试集

            dataAugment(train_list, test_list, trainPath, testPath, defectPath)

def dataAugment(train_list, test_list, trainPath, testPath, defectPath):
    for file in train_list:
        imageName = file.split('.')[0]
        imgpath = os.path.join(defectPath,file)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

        # if 'pressing' in defectPath or 'Zinc residue' in defectPath:
        #     img = centerCrop(img)

        # resize到统一的200*200
        img = cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC).copy()

        # 裁剪
        img_cropped1 = img[0:166,0:166]
        img_cropped2 = img[0:166, 34:200]
        img_cropped3 = img[34:200, 0:166]
        img_cropped4 = img[34:200, 34:200]
        img_cropped5 = img[17:183, 17:183]

        # 水平翻转
        img_flapped1 = cv2.flip(img_cropped1, 1, dst=None)
        img_flapped2 = cv2.flip(img_cropped2, 1, dst=None)
        img_flapped3 = cv2.flip(img_cropped3, 1, dst=None)
        img_flapped4 = cv2.flip(img_cropped4, 1, dst=None)
        img_flapped5 = cv2.flip(img_cropped5, 1, dst=None)

        # 均值模糊和高斯模糊
        Blur = cv2.blur(img, (5, 5), 3)
        Gussian_blur = cv2.GaussianBlur(img, (5,5), 3)

        SaltNoise = SaltAndPepper(img, 0.1)
        GussianNoise = addGussianNoise(img, 0.1)

        RotateMatrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=180, scale=1)
        RotImg_180 = cv2.warpAffine(img, RotateMatrix, (img.shape[0], img.shape[1]))

        img_aug = [img_cropped1, img_cropped2, img_cropped3, img_cropped4, img_cropped5,
                   img_flapped1, img_flapped2, img_flapped3, img_flapped4, img_flapped5,
                   Blur, Gussian_blur, SaltNoise, GussianNoise, RotImg_180]

        augImgList = []   #(image,name)
        for i in range(len(img_aug)):
            newName = imageName + '_aug_' + str(i+1) + '.bmp'
            savePath = os.path.join(trainPath, newName)
            img_aug[i] = cv2.resize(img_aug[i],(224, 224))
            cv2.imwrite(savePath, img_aug[i])

    for file in test_list:
        imgPath = os.path.join(defectPath, file)
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # if 'pressing' in defectPath  or 'Zinc residue' in defectPath:
        #     img = centerCrop(img)
        img = cv2.resize(img, (224, 224))
        savePath = os.path.join(testPath, file)
        cv2.imwrite(savePath, img)

def centerCrop(img):
    height, width = img.shape[0], img.shape[1]
    left = int(width / 4)
    top = int(height / 4)
    img1 = img[top:int(3 * height / 4), left:int(3 * width / 4)].copy()
    return img1

# 椒盐噪声
def SaltAndPepper(img, percentage):
    src = img.copy()
    SP_NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.randint(0,1) == 0:
            src[randX,randY] = 0
        else:
            src[randX, randY] = 255
    return src

# 高斯噪声
def addGussianNoise(img, percentage):
    src = img.copy()
    G_NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(G_NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        src[randX,randY] = 255
    return src

if __name__ == '__main__':
    # do_process()
    do_dataAugment()
    print('')



