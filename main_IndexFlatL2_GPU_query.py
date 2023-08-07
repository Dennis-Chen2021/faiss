import os
import faiss
import torch
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import time

# GPU
res = faiss.StandardGpuResources()

# 加载预训练的 ResNet50 模型，并去掉全连接层
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval()

# 移到GPU
device = torch.device("cuda")
resnet50.to(device)

# 预处理
preprocessor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# 加载 Faiss 索引和图像名称字典
indexname = "index_9292_IndexFlatL2_GPU.index"
index_cpu = faiss.read_index(indexname)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # 将索引移动到 GPU 上
image_dict = dict(np.load(indexname + ".npy"))

filename_list = list(image_dict.values())


# 特征提取
def extract_features(image_path):
    image = Image.open(image_path)
    tensor = preprocessor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet50(tensor).squeeze()
        feature = torch.nn.functional.normalize(feature, p=2, dim=0)
        feature = feature.cpu().numpy()
    return feature


# faiss
def search_faiss(query_feature, index, k=5):
    D, I = index.search(np.array([query_feature]), k)
    results = []
    for i, distance in zip(I[0], D[0]):
        filename = filename_list[i]
        results.append((i, filename, distance))
    return results


# Faiss GPU search
def search_index(query_path, k=5):
    # 提取查询图像的特征向量
    start_feature_time = time.perf_counter()
    query_feature = extract_features(query_path)
    end_feature_time = time.perf_counter()
    feature_time = end_feature_time - start_feature_time
    
    # 在 Faiss GPU 索引中查找相似图像
    start_search_time = time.perf_counter()
    results = search_faiss(query_feature, index_gpu, k)
    end_search_time = time.perf_counter()
    search_time = end_search_time - start_search_time
    
    return results, search_time, feature_time


def main():
    folder_path = './query'
    k = 5
    results_dict = {}  # a dictionary to store the results for each image
    search_time_total = 0;
    num_image = 0
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.jpg'):
            continue
        query_path = os.path.join(folder_path, filename)
        results, search_time, feature_time = search_index(query_path, k)
        results_dict[filename] = results
        print(f"Query image: {query_path}")
        for i, filename, distance in results:
            print(f"Index: {i}, Filename: {filename}, Distance: {distance:.5f}")
        print(f"Feature extraction time: {feature_time*1000:.5f} milliseconds")
        print(f"Search time: {search_time*1000:.5f} milliseconds")
        search_time_total = search_time_total +  search_time
        num_image +=1
        
    search_time_avg = search_time_total/num_image
    print(f"search_time_avg time: {search_time_avg*1000:.5f} milliseconds")


if __name__ == '__main__':
    main()