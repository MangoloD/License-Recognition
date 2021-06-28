import os
import torch
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
from cnn2seq import MainNet
from utils import label_to_str

img_dir = "../fake_chs_lp/images"
img_names = os.listdir(img_dir)
net = MainNet()
net.load_state_dict(torch.load(r"../params/cnn2seq.pth"))
net.eval()

trans = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])
l1 = ['新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', '吉', '闽', '贵', '粤', '青',
      '藏', '川', '宁', '琼', '辽', '黑', '湘', '皖', '鲁', '京', '津', '沪', '渝', '冀', '豫', '云']
l2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
      'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

with torch.no_grad():
    for i in range(len(img_names)):
        img_path = os.path.join(img_dir, img_names[i])
        img = Image.open(img_path)
        data = trans(img).unsqueeze(0)
        out_ = net(data)
        out = torch.argmax(out_, 2).numpy()[0]

        x = label_to_str(out)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(x, fontdict={'family': 'SimHei', 'size': 20})
plt.show()
