import os
import torch
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from net import MainNet

img_dir = "G:/DL_Data/Plate/test_plate"
img_names = os.listdir(img_dir)
net = MainNet()
net.load_state_dict(torch.load(r"../params/seq2seq.pth"))
net.eval()

trans = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])
l1 = ['新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', '吉', '闽', '贵', '粤', '青',
      '藏', '川', '宁', '琼', '辽', '黑', '湘', '皖', '鲁', '京', '津', '沪', '渝', '冀', '豫', '云']
l2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
      'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
a = np.random.randint(0, len(img_names), 9)
with torch.no_grad():
    for i in range(9):
        img_path = os.path.join(img_dir, img_names[a[i]])
        img = Image.open(img_path)
        data = trans(img).unsqueeze(0)
        out = net(data).squeeze()
        out = torch.argmax(out, 1).numpy()
        predict = l1[out[0]]
        for idx in out[1:]:
            predict += l2[idx]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(predict, fontdict={'family': 'SimHei', 'size': 20})
plt.show()
