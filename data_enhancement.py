import os
import random
from PIL import Image, ImageOps, ImageEnhance

# 定义数据增强函数
def data_augmentation(image):
    # 随机旋转
    angle = random.randint(-10, 10)
    image = image.rotate(angle)

    # 随机裁剪和缩放
    width, height = image.size
    crop_size = random.randint(int(min(width, height) * 0.8), int(min(width, height) * 0.9))
    crop_left = random.randint(0, width - crop_size)
    crop_top = random.randint(0, height - crop_size)
    crop_box = (crop_left, crop_top, crop_left + crop_size, crop_top + crop_size)
    image = image.crop(crop_box)
    image = image.resize((224, 224))

    # 随机水平翻转
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # 随机亮度、对比度和色彩饱和度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    return image

# 原始图片所在的文件夹
input_dir = './tmp'

# 增强后的图片保存的文件夹
output_dir = './tmp_output100'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历原始图片所在的文件夹中的所有图片
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        # 加载图片
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # 对同一张图片进行百次不同的数据增强
        for i in range(100):
            # 对图片进行数据增强
            img_aug = data_augmentation(img)

            # 将增强后的图片保存到不同的文件中
            output_path = os.path.join(output_dir, '{}_{}.jpg'.format(os.path.splitext(filename)[0], i+1))
            img_aug.save(output_path)