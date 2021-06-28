provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
             "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
             "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
       'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

"""
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

025:Area: Area ratio of license plate area to the entire picture area.

95_113: Tilt degree: Horizontal tilt degree and vertical tilt degree.

154&383_386&473: Bounding box coordinates: The coordinates of the left-up and the right-bottom vertices.

386&473_177&454_154&383_363&402: Four vertices locations: The exact (x, y) coordinates of the four vertices of LP 
                                 in the whole image. These coordinates start from the right-bottom vertex.
                                 
0_0_22_27_27_33_16: License plate number: province (1 character), 
                                          alphabets (1 character), 
                                          alphabets+digits (5 characters).
                                          
37: Brightness: The brightness of the license plate region.

15: Blurriness: The Blurriness of the license plate region.
"""

upper = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J', 19: 'K',
         20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
         30: 'W', 31: 'X', 32: 'Y', 33: 'Z'}

province = {34: '藏', 35: '川', 36: '鄂', 37: '甘', 38: '赣', 39: '贵', 40: '桂', 41: '黑', 42: '沪',
            43: '吉', 44: '冀', 45: '津', 46: '晋', 47: '京', 48: '辽', 49: '鲁', 50: '蒙', 51: '闽',
            52: '宁', 53: '青', 54: '琼', 55: '陕', 56: '苏', 57: '皖', 58: '湘', 59: '新', 60: '渝',
            61: '豫', 62: '粤', 63: '云', 64: '浙'}


# 字符转为分类类别
def str_to_label(string):
    label = []
    for i in range(7):
        # 0-9
        if 48 <= ord(string[i]) <= 57:
            label.append(ord(string[i]) - ord('0'))
        # A-Z
        elif 65 <= ord(string[i]) <= 90:
            for num, value in upper.items():
                if string[i] == value:
                    label.append(num)
        # 省份
        else:
            for num, value in province.items():
                if string[i] == value:
                    label.append(num)
    return label


def label_to_str(label):
    string = ""
    for index in label:
        if index <= 9:
            string += str(index)
        elif 10 <= index <= 33:
            for num, value in upper.items():
                if num == index:
                    string += value
            # 省份
        else:
            for num, value in province.items():
                if num == index:
                    string += value
    return string


if __name__ == '__main__':
    label_ = str_to_label("桂4F4HJT")
    print(label_)
    print(label_to_str(label_))
