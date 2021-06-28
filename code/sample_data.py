import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import str_to_label

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class Sampling(Dataset):
    def __init__(self, root):
        self.transform = data_transforms
        self.images = []
        self.labels = []

        for filename in os.listdir(root):
            x = os.path.join(root, filename)
            y = filename.split(".")[0]
            self.images.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = self.transform(Image.open(image_path))
        label = self.labels[index]
        label = str_to_label(label)  # 将字母转成数字表示，方便做one-hot
        label = self.one_hot(label)
        # label = torch.Tensor(label)

        return image, label

    @staticmethod
    def one_hot(x):
        z = np.zeros((7, 65))
        for i in range(7):
            index = int(x[i])
            z[i][index] = 1
        return z


if __name__ == '__main__':
    sampling = Sampling("G:/DL_Data/Plate/train_plate")
    dataloader = DataLoader(sampling, 10, shuffle=True)
    for j, (img, labels) in enumerate(dataloader):
        # print(img.shape)
        print(labels)
        print(labels.shape)
        exit()
