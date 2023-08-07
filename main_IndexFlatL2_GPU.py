import os
import faiss
import torch
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import time

def main():
    # GPU
    res = faiss.StandardGpuResources()

    # 加载预训练的 ResNet50 模型，并去掉全连接层
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    resnet50.eval()

    # 将模型移动到 GPU 上
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

    # 创建 Faiss 索引和图像名称字典
    root = './tmp_output100'
    features_list = []
    image_names = []
    start_time = time.perf_counter() 
    for i, filename in enumerate(os.listdir(root)):
        if filename.endswith('.jpg'):
            img_path = os.path.join(root, filename)
            img = Image.open(img_path)
            img_tensor = preprocessor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = resnet50(img_tensor).squeeze()
                features = torch.nn.functional.normalize(features, p=2, dim=0)
                features = features.cpu().numpy()
                features_list.append(features)
                image_names.append(filename)
    end_time = time.perf_counter() 
    num_images = len(features_list)
    print(f"Extracting features from {num_images} images took {(end_time - start_time) * 1000:.2f} ms.")

    d = 2048  # 特征向量维度
    index_flat = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    features_array = np.array(features_list).astype('float32')

    start_time = time.perf_counter() 
    index.add(features_array)
    end_time = time.perf_counter() 
    print(f"Adding {num_images} feature vectors to index took {(end_time - start_time) * 1000:.2f} ms.")

    # 将 GPU 索引转换为 CPU 索引
    index_cpu = faiss.index_gpu_to_cpu(index)

    # 将 Faiss 索引和图像名称字典保存到磁盘中
    start_time = time.perf_counter()
    indexname = "index_"+str(num_images)+"_IndexFlatL2_GPU.index"
    faiss.write_index(index_cpu, indexname)
    np.save(indexname + ".npy", np.array(list(enumerate(image_names))))
    end_time = time.perf_counter()
    print(f"Saving index and image names to disk took {(end_time - start_time) * 1000:.2f} ms.")

if __name__ == '__main__':
    main()