import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import time

# ResNet-50
resnet = models.resnet50(pretrained=True)
resnet.eval()

# GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 提取特征
def extract_features(image_name):
    image = Image.open(image_name)
    image_tensor = transform(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features = resnet(image_tensor)
    features = torch.flatten(features)
    return features.cpu().numpy()

# L2距离
def distance(feature1, feature2):
    diff = feature1 - feature2
    return torch.sqrt(torch.sum(torch.pow(diff, 2)))

# 查找最相似的图像
def find_similar_images(query_image, tmp_features):
    query_feature = extract_features(query_image)
    
    similarities = []
    # 遍历所有tmp图像并计算L2距离
    start_time = time.perf_counter()
    for tmp_image, tmp_feature in tmp_features.items():
        dist = distance(torch.from_numpy(query_feature), torch.from_numpy(tmp_feature))
        similarities.append((tmp_image, dist))
    # 按相似度排序
    similarities.sort(key=lambda x: x[1])
    end_time = time.perf_counter()
    # 返回top 5最相似的图像和查询时间
    return [x[0] for x in similarities[:5]], end_time - start_time

def main():
    # 加载tmp文件夹中的所有图像并提取特征
    tmp_features = {}
    start_time_extract_features = time.perf_counter()
    for filename in os.listdir("tmp_output100"):
        if filename.endswith(".jpg"):
            tmp_image = os.path.join("tmp_output100", filename)
            tmp_features[tmp_image] = extract_features(tmp_image)
    end_time_extract_features = time.perf_counter()
    
    print(f"extract_features time: {(end_time_extract_features-start_time_extract_features)*1000:.5f} ms")

    num_image = 0
    search_time_total = 0

    # 加载query文件夹中的所有图像并查找最相似的图像
    for filename in os.listdir("query"):
        if filename.endswith(".jpg"):
            query_image = os.path.join("query", filename)
            similar_images, query_time = find_similar_images(query_image, tmp_features)
            print(f"{filename}:\n {similar_images} \n ({query_time * 1000:.5f} ms)")
            search_time_total = search_time_total + query_time
            num_image = num_image + 1
            print("num_image:",num_image)

    # 计算平均查询时间
    search_time_avg = search_time_total / num_image * 1000
    print(f"search_time_avg time: {search_time_avg:.5f} ms")

if __name__ == '__main__':
    main()