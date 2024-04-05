import torch
import numpy as np
from PIL import Image

image_list = ["000000000009", "000000027550", "000000218668"]

for img in image_list:
    image_path = "../../../../mnt/sda/suhohan/coco2017/train2017/{}.jpg".format(img)

    # 이미지 불러오기
    image = Image.open(image_path)

    # 이미지 크기 확인
    width, height = image.size
    print("이미지 너비:", width)
    print("이미지 높이:", height)

    # PIL 이미지를 NumPy 배열로 변환
    image_array = np.array(image)

    # NumPy 배열을 PyTorch 텐서로 변환
    image_tensor = torch.from_numpy(image_array)

    # 텐서 모양 확인
    print("이미지 텐서 모양:", image_tensor.shape)
