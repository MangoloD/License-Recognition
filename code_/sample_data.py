import os
import torch
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class Sampling(data.Dataset):
    def __init__(self, root, is_train):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.images = []
        self.labels = []
        for filename in os.listdir(root):
            x = os.path.join(root, filename)
            y = filename.split(".")[0]
            self.images.append(x)
            self.labels.append(y)
        self.l1 = ['新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', '吉', '闽', '贵', '粤', '青',
                   '藏', '川', '宁', '琼', '辽', '黑', '湘', '皖', '鲁', '京', '津', '沪', '渝', '冀', '豫', '云']
        self.l2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                   'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.labels[index]
        label = self.one_hot(label)
        return img, label

    def one_hot(self, x):
        z = torch.zeros(7, 36)
        index = self.l1.index(x[0])
        z[0][index] = 1
        for i in range(1, 7):
            index = self.l2.index(x[i])
            z[i][index] = 1
        return z
