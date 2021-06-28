from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
import numpy as np


# 随机字母
def ran_num():
    a = str(random.randint(0, 9))
    a = chr(random.randint(48, 57))
    b = chr(random.randint(65, 90))  # 大写字母
    c = chr(random.randint(97, 122))  # 小写字母
    d = ord(c)
    # print(c)
    # print(d)
    # exit()
    return a


# 随机颜色
def ran_color():
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255))


# 240*60
w = 240
h = 60
# 创建字体对象
font = ImageFont.truetype("arial.ttf", 40)
for i in range(100):
    # 随机生成像素矩阵
    img_arr = np.random.randint(128, 255, (h, w, 3))
    # 将像素矩阵转换成背景图片
    image = Image.fromarray(np.uint8(img_arr))
    # 创建Draw对象
    draw = ImageDraw.Draw(image)
    # 填充文字像素颜色
    filename = ""
    for j in range(4):
        ch = ran_num()
        filename += ch
        # 给字体之间增加间隔
        draw.text((40 * j + 20 * (j + 1), 10), ch, font=font, fill=ran_color())

    # 模糊:
    # image = image.filter(ImageFilter.BLUR)
    # image.show()
    if not os.path.exists("images"):
        os.makedirs("images")
    image_path = r"images"
    image.save("{0}/{1}.jpg".format(image_path, filename))
    print(i)
