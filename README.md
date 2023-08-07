

# 卡车图片--faiss检索Demo总结

---

## 0.硬件配置

> CPU: AMD Ryzen 7 5800H with Radeon Graphics 
>
> GPU: GTX 1650 



## 1.采用faiss比较

前言：考虑到所给图片过少，不能体现是faiss优势来，所以先将图片进行数据增强得到9292张图片，达到近万级别数据；

(将数据集扩充100倍，实际操作中不会扩充这么多倍，此处只为了验证faiss检索)

resnet50 去除全连接层 -- 特征向量 2048维度

下表为暴力检索和faiss的几种索引的实际测试效果:

|                      方法                       |              优缺点               |                             缺点                             | 数据量级 | 检索时间(ms) 20张图片平均 |
| :---------------------------------------------: | :-------------------------------: | :----------------------------------------------------------: | :------: | :-----------------------: |
|                  遍历暴力检索                   |               准确                |                          慢，极其慢                          |   9292   |       **292.37646**       |
|   Faiss--IndexFlatL2 (适合数据量**50万**以内)   |               准确                |                     检索速度慢，占内存大                     |   9292   |          1.92864          |
|    Faiss--IndexIVFFlat(适合数据量**百万**级)    | 检索快于IndexFlatL2，内存消耗不大 |                         数据需要训练                         |   9292   |        **0.65775**        |
| Faiss--IndexIVFPQ(适合一百万到**一亿**的数据量) |          高效的查询速度           | 索引构建时间长，需要调参到达最佳性能(调参比较困难)、对数据集要求较高 |   9292   |          1.10738          |
|  Faiss--IndexHNSWFlat(适合数据量**百亿**级别)   |            不需要训练             |                  构建索引极慢，占用内存极大                  |   9292   |    /(电脑跑不动，故略)    |

注:IndexIVFPQ需要调参方能达到最佳，此处聚类中心数量设置为100，探测数设置为8

> 较暴力检索而言，faiss在现有条件下，快400多倍

## 2. faiss 索引类型

### 2.1 暴力检索

```bash
search_time_avg time: 292.37646 ms
```



### 2.2 IndexFlatL2

> 优点：该方法是Faiss所有index中最准确的，召回率最高的方法，没有之一；
>
> 缺点：检索速度慢，占内存大，构建索引快。（不需要训练，可以逐条添加数据）
>
> 使用情况：向量候选集很少，数据量在50万以内，可以考虑使用内存索引，在普通的 PC 机上检索时间在 < 10ms 的级别

2.2.1 抽特征，得索引

```bash
root@d127cfa55267:/workspace/faiss_test_car# python main_IndexFlatL2_GPU.py
Extracting features from 9292 images took 208558.66 ms.
Adding 9292 feature vectors to index took 22.79 ms.
Saving index and image names to disk took 136.41 ms.
```

2.2.2 据索引查询

```
root@d127cfa55267:/workspace/faiss_test_car# python main_IndexFlatL2_GPU_query.py
Query image: ./query/20190525_0000_44c5974b76.jpg
Index: 0, Filename: 20190525_0000_44c5974b76.jpg, Distance: 0.00000
Index: 51, Filename: 20190525_0000_44c5974b76_54.jpg, Distance: 0.12454
Index: 7, Filename: 20190525_0000_44c5974b76_14.jpg, Distance: 0.13724
Index: 71, Filename: 20190525_0000_44c5974b76_72.jpg, Distance: 0.14616
Index: 50, Filename: 20190525_0000_44c5974b76_53.jpg, Distance: 0.14914
Feature extraction time: 772.74727 milliseconds
Search time: 2.47625 milliseconds
Query image: ./query/20190525_0001_814ed64fa4.jpg
Index: 101, Filename: 20190525_0001_814ed64fa4.jpg, Distance: 0.00000
Index: 120, Filename: 20190525_0001_814ed64fa4_25.jpg, Distance: 0.12375
Index: 179, Filename: 20190525_0001_814ed64fa4_79.jpg, Distance: 0.12689
Index: 158, Filename: 20190525_0001_814ed64fa4_6.jpg, Distance: 0.12814
Index: 157, Filename: 20190525_0001_814ed64fa4_59.jpg, Distance: 0.13268
Feature extraction time: 23.40482 milliseconds
Search time: 2.29255 milliseconds
Query image: ./query/20190525_0002_587962ae63.jpg
Index: 202, Filename: 20190525_0002_587962ae63.jpg, Distance: 0.00000
Index: 7171, Filename: 20190525_0071_3d247afb52.jpg, Distance: 0.06479
Index: 243, Filename: 20190525_0002_587962ae63_45.jpg, Distance: 0.10974
Index: 220, Filename: 20190525_0002_587962ae63_24.jpg, Distance: 0.11126
Index: 228, Filename: 20190525_0002_587962ae63_31.jpg, Distance: 0.11782
Feature extraction time: 22.38951 milliseconds
Search time: 2.15736 milliseconds
Query image: ./query/20190525_0003_d427e73752.jpg
Index: 303, Filename: 20190525_0003_d427e73752.jpg, Distance: 0.00000
Index: 361, Filename: 20190525_0003_d427e73752_60.jpg, Distance: 0.12136
Index: 401, Filename: 20190525_0003_d427e73752_97.jpg, Distance: 0.13318
Index: 376, Filename: 20190525_0003_d427e73752_74.jpg, Distance: 0.13469
Index: 304, Filename: 20190525_0003_d427e73752_1.jpg, Distance: 0.14531
Feature extraction time: 23.47312 milliseconds
Search time: 2.15352 milliseconds
Query image: ./query/20190525_0004_668fd3b75c.jpg
Index: 404, Filename: 20190525_0004_668fd3b75c.jpg, Distance: 0.00000
Index: 453, Filename: 20190525_0004_668fd3b75c_52.jpg, Distance: 0.14267
Index: 443, Filename: 20190525_0004_668fd3b75c_43.jpg, Distance: 0.14903
Index: 430, Filename: 20190525_0004_668fd3b75c_31.jpg, Distance: 0.16063
Index: 440, Filename: 20190525_0004_668fd3b75c_40.jpg, Distance: 0.16140
Feature extraction time: 24.03542 milliseconds
Search time: 2.24920 milliseconds
Query image: ./query/20190525_0005_32d3b42dee.jpg
Index: 505, Filename: 20190525_0005_32d3b42dee.jpg, Distance: 0.00000
Index: 546, Filename: 20190525_0005_32d3b42dee_45.jpg, Distance: 0.11169
Index: 507, Filename: 20190525_0005_32d3b42dee_10.jpg, Distance: 0.11401
Index: 528, Filename: 20190525_0005_32d3b42dee_29.jpg, Distance: 0.12564
Index: 535, Filename: 20190525_0005_32d3b42dee_35.jpg, Distance: 0.12807
Feature extraction time: 24.30182 milliseconds
Search time: 2.21104 milliseconds
Query image: ./query/20190525_0006_19969d7a10.jpg
Index: 606, Filename: 20190525_0006_19969d7a10.jpg, Distance: 0.00000
Index: 645, Filename: 20190525_0006_19969d7a10_43.jpg, Distance: 0.10394
Index: 676, Filename: 20190525_0006_19969d7a10_71.jpg, Distance: 0.11919
Index: 672, Filename: 20190525_0006_19969d7a10_68.jpg, Distance: 0.12018
Index: 657, Filename: 20190525_0006_19969d7a10_54.jpg, Distance: 0.12137
Feature extraction time: 21.50844 milliseconds
Search time: 2.26411 milliseconds
Query image: ./query/20190525_0007_b8fc3fd5d6.jpg
Index: 707, Filename: 20190525_0007_b8fc3fd5d6.jpg, Distance: 0.00000
Index: 783, Filename: 20190525_0007_b8fc3fd5d6_77.jpg, Distance: 0.09123
Index: 5050, Filename: 20190525_0050_03c42b6967.jpg, Distance: 0.09191
Index: 711, Filename: 20190525_0007_b8fc3fd5d6_11.jpg, Distance: 0.10301
Index: 5127, Filename: 20190525_0050_03c42b6967_78.jpg, Distance: 0.10349
Feature extraction time: 22.87407 milliseconds
Search time: 2.21187 milliseconds
Query image: ./query/20190525_0008_37270aae38.jpg
Index: 808, Filename: 20190525_0008_37270aae38.jpg, Distance: 0.00000
Index: 904, Filename: 20190525_0008_37270aae38_95.jpg, Distance: 0.08558
Index: 894, Filename: 20190525_0008_37270aae38_86.jpg, Distance: 0.09968
Index: 841, Filename: 20190525_0008_37270aae38_38.jpg, Distance: 0.10723
Index: 831, Filename: 20190525_0008_37270aae38_29.jpg, Distance: 0.10790
Feature extraction time: 25.52032 milliseconds
Search time: 2.46872 milliseconds
Query image: ./query/20190525_0009_7c3de64969.jpg
Index: 909, Filename: 20190525_0009_7c3de64969.jpg, Distance: 0.00000
Index: 975, Filename: 20190525_0009_7c3de64969_68.jpg, Distance: 0.09462
Index: 8888, Filename: 20190525_0088_57f3d2be4b.jpg, Distance: 0.09595
Index: 953, Filename: 20190525_0009_7c3de64969_48.jpg, Distance: 0.11713
Index: 917, Filename: 20190525_0009_7c3de64969_15.jpg, Distance: 0.11743
Feature extraction time: 24.25340 milliseconds
Search time: 1.66912 milliseconds
Query image: ./query/20190525_0082_05b34dfe35.jpg
Index: 8282, Filename: 20190525_0082_05b34dfe35.jpg, Distance: 0.00000
Index: 4848, Filename: 20190525_0048_d3cddd9ef3.jpg, Distance: 0.09383
Index: 2323, Filename: 20190525_0023_4e3a2535c8.jpg, Distance: 0.11490
Index: 2359, Filename: 20190525_0023_4e3a2535c8_40.jpg, Distance: 0.11919
Index: 2525, Filename: 20190525_0025_3acff8485b.jpg, Distance: 0.12424
Feature extraction time: 31.37181 milliseconds
Search time: 1.82044 milliseconds
Query image: ./query/20190525_0083_5897e1c17b.jpg
Index: 8383, Filename: 20190525_0083_5897e1c17b.jpg, Distance: 0.00000
Index: 8468, Filename: 20190525_0083_5897e1c17b_85.jpg, Distance: 0.09815
Index: 8438, Filename: 20190525_0083_5897e1c17b_58.jpg, Distance: 0.10001
Index: 8456, Filename: 20190525_0083_5897e1c17b_74.jpg, Distance: 0.10044
Index: 8448, Filename: 20190525_0083_5897e1c17b_67.jpg, Distance: 0.10328
Feature extraction time: 27.40041 milliseconds
Search time: 1.65103 milliseconds
Query image: ./query/20190525_0084_e104a84e60.jpg
Index: 8484, Filename: 20190525_0084_e104a84e60.jpg, Distance: 0.00000
Index: 8522, Filename: 20190525_0084_e104a84e60_42.jpg, Distance: 0.21986
Index: 8543, Filename: 20190525_0084_e104a84e60_61.jpg, Distance: 0.23164
Index: 8506, Filename: 20190525_0084_e104a84e60_28.jpg, Distance: 0.23545
Index: 8487, Filename: 20190525_0084_e104a84e60_100.jpg, Distance: 0.23896
Feature extraction time: 25.39318 milliseconds
Search time: 1.56450 milliseconds
Query image: ./query/20190525_0085_9c3c6bc7d8.jpg
Index: 8585, Filename: 20190525_0085_9c3c6bc7d8.jpg, Distance: 0.00000
Index: 8620, Filename: 20190525_0085_9c3c6bc7d8_4.jpg, Distance: 0.14355
Index: 8663, Filename: 20190525_0085_9c3c6bc7d8_79.jpg, Distance: 0.16225
Index: 8632, Filename: 20190525_0085_9c3c6bc7d8_50.jpg, Distance: 0.17066
Index: 8595, Filename: 20190525_0085_9c3c6bc7d8_17.jpg, Distance: 0.17368
Feature extraction time: 23.63441 milliseconds
Search time: 1.59730 milliseconds
Query image: ./query/20190525_0086_4cd57b6244.jpg
Index: 8686, Filename: 20190525_0086_4cd57b6244.jpg, Distance: 0.00000
Index: 8762, Filename: 20190525_0086_4cd57b6244_77.jpg, Distance: 0.10554
Index: 8757, Filename: 20190525_0086_4cd57b6244_72.jpg, Distance: 0.12015
Index: 8717, Filename: 20190525_0086_4cd57b6244_36.jpg, Distance: 0.12471
Index: 8722, Filename: 20190525_0086_4cd57b6244_40.jpg, Distance: 0.12905
Feature extraction time: 21.82690 milliseconds
Search time: 1.75501 milliseconds
Query image: ./query/20190525_0087_dc908ca4d5.jpg
Index: 8787, Filename: 20190525_0087_dc908ca4d5.jpg, Distance: 0.00000
Index: 8805, Filename: 20190525_0087_dc908ca4d5_24.jpg, Distance: 0.08822
Index: 8790, Filename: 20190525_0087_dc908ca4d5_100.jpg, Distance: 0.10004
Index: 8808, Filename: 20190525_0087_dc908ca4d5_27.jpg, Distance: 0.11150
Index: 8871, Filename: 20190525_0087_dc908ca4d5_84.jpg, Distance: 0.11718
Feature extraction time: 23.10313 milliseconds
Search time: 1.58827 milliseconds
Query image: ./query/20190525_0088_57f3d2be4b.jpg
Index: 8888, Filename: 20190525_0088_57f3d2be4b.jpg, Distance: 0.00000
Index: 909, Filename: 20190525_0009_7c3de64969.jpg, Distance: 0.09595
Index: 994, Filename: 20190525_0009_7c3de64969_85.jpg, Distance: 0.10479
Index: 917, Filename: 20190525_0009_7c3de64969_15.jpg, Distance: 0.10644
Index: 934, Filename: 20190525_0009_7c3de64969_30.jpg, Distance: 0.12177
Feature extraction time: 24.20758 milliseconds
Search time: 1.64543 milliseconds
Query image: ./query/20190525_0089_a2c93bd374.jpg
Index: 8989, Filename: 20190525_0089_a2c93bd374.jpg, Distance: 0.00000
Index: 3535, Filename: 20190525_0035_67b9426ac9.jpg, Distance: 0.04772
Index: 3620, Filename: 20190525_0035_67b9426ac9_85.jpg, Distance: 0.09340
Index: 9003, Filename: 20190525_0089_a2c93bd374_20.jpg, Distance: 0.10335
Index: 3555, Filename: 20190525_0035_67b9426ac9_26.jpg, Distance: 0.10353
Feature extraction time: 23.75058 milliseconds
Search time: 1.58089 milliseconds
Query image: ./query/20190525_0090_4941737983.jpg
Index: 9090, Filename: 20190525_0090_4941737983.jpg, Distance: 0.00000
Index: 9120, Filename: 20190525_0090_4941737983_35.jpg, Distance: 0.09463
Index: 9186, Filename: 20190525_0090_4941737983_95.jpg, Distance: 0.09667
Index: 9142, Filename: 20190525_0090_4941737983_55.jpg, Distance: 0.12224
Index: 9145, Filename: 20190525_0090_4941737983_58.jpg, Distance: 0.12481
Feature extraction time: 24.83042 milliseconds
Search time: 1.59533 milliseconds
Query image: ./query/20190525_0091_71be77efe3.jpg
Index: 9191, Filename: 20190525_0091_71be77efe3.jpg, Distance: 0.00000
Index: 9286, Filename: 20190525_0091_71be77efe3_94.jpg, Distance: 0.07018
Index: 9198, Filename: 20190525_0091_71be77efe3_14.jpg, Distance: 0.10283
Index: 9261, Filename: 20190525_0091_71be77efe3_71.jpg, Distance: 0.10884
Index: 9209, Filename: 20190525_0091_71be77efe3_24.jpg, Distance: 0.10900
Feature extraction time: 25.64961 milliseconds
Search time: 1.62095 milliseconds
search_time_avg time: 1.92864 milliseconds
```



-----

### 2.3 IndexIVFFlat

> 倒排暴力检索 Inverted Flat
>
> 优点：检索的性能快于暴力检索，内存消耗不大
>
> 缺点：数据需要训练，需要批量训练和一次性批量插入（训练的数据跟插入的集合是相同的集合，不合适插入新的数据，创建的索引适合用作不变的数据集） IVF利用近似于倒排索引的思想，对训练集做聚类之后，数据被划分到 n 个聚类中心下。检索时， 计算每个聚类中心的向量跟目标向量的距离，找到 p 个最相近的聚类中心，然后再检索各个聚类中心附近的所有非中心点的距离，从而找出最近的 k 个点。IVF 通过减小搜索范围，提升了搜索效率。
>
> 对于百万的数据，通常可以采用 4096, 或者 16384 个聚类中心的 IVF_Flat 索引。具体的参数根据你的数据集做相应的 召回率， 准确率和检索性能的测试，最终选择合适的参数。




2.3.1 抽特征，得索引

```bash
root@d127cfa55267:/workspace/faiss_test_car# python main_IndexIVFFlat_GPU.py
Extracting features from 9292 images took 157210.68 ms.
Training and adding 9292 feature vectors to index took 311.56 ms.
Saving index and image names to disk took 137.72 ms.
```

2.3.2 据索引查询

```bash
root@d127cfa55267:/workspace/faiss_test_car# python main_IndexIVFFlat_GPU_query.py
Query image: ./query/20190525_0000_44c5974b76.jpg
Index: 0, Filename: 20190525_0000_44c5974b76.jpg, Distance: 0.00000
Index: 51, Filename: 20190525_0000_44c5974b76_54.jpg, Distance: 0.12454
Index: 7, Filename: 20190525_0000_44c5974b76_14.jpg, Distance: 0.13724
Index: 71, Filename: 20190525_0000_44c5974b76_72.jpg, Distance: 0.14616
Index: 50, Filename: 20190525_0000_44c5974b76_53.jpg, Distance: 0.14915
Feature extraction time: 732.58965 milliseconds
Search time: 0.77652 milliseconds
Query image: ./query/20190525_0001_814ed64fa4.jpg
Index: 101, Filename: 20190525_0001_814ed64fa4.jpg, Distance: 0.00000
Index: 120, Filename: 20190525_0001_814ed64fa4_25.jpg, Distance: 0.12375
Index: 179, Filename: 20190525_0001_814ed64fa4_79.jpg, Distance: 0.12689
Index: 157, Filename: 20190525_0001_814ed64fa4_59.jpg, Distance: 0.13268
Index: 178, Filename: 20190525_0001_814ed64fa4_78.jpg, Distance: 0.14291
Feature extraction time: 27.67712 milliseconds
Search time: 0.70785 milliseconds
Query image: ./query/20190525_0002_587962ae63.jpg
Index: 202, Filename: 20190525_0002_587962ae63.jpg, Distance: 0.00000
Index: 7171, Filename: 20190525_0071_3d247afb52.jpg, Distance: 0.06479
Index: 243, Filename: 20190525_0002_587962ae63_45.jpg, Distance: 0.10974
Index: 220, Filename: 20190525_0002_587962ae63_24.jpg, Distance: 0.11126
Index: 228, Filename: 20190525_0002_587962ae63_31.jpg, Distance: 0.11782
Feature extraction time: 26.23897 milliseconds
Search time: 0.54165 milliseconds
Query image: ./query/20190525_0003_d427e73752.jpg
Index: 303, Filename: 20190525_0003_d427e73752.jpg, Distance: 0.00000
Index: 361, Filename: 20190525_0003_d427e73752_60.jpg, Distance: 0.12136
Index: 401, Filename: 20190525_0003_d427e73752_97.jpg, Distance: 0.13319
Index: 376, Filename: 20190525_0003_d427e73752_74.jpg, Distance: 0.13469
Index: 304, Filename: 20190525_0003_d427e73752_1.jpg, Distance: 0.14531
Feature extraction time: 23.97076 milliseconds
Search time: 0.64494 milliseconds
Query image: ./query/20190525_0004_668fd3b75c.jpg
Index: 404, Filename: 20190525_0004_668fd3b75c.jpg, Distance: 0.00000
Index: 453, Filename: 20190525_0004_668fd3b75c_52.jpg, Distance: 0.14267
Index: 443, Filename: 20190525_0004_668fd3b75c_43.jpg, Distance: 0.14903
Index: 430, Filename: 20190525_0004_668fd3b75c_31.jpg, Distance: 0.16063
Index: 440, Filename: 20190525_0004_668fd3b75c_40.jpg, Distance: 0.16140
Feature extraction time: 25.06377 milliseconds
Search time: 0.63778 milliseconds
Query image: ./query/20190525_0005_32d3b42dee.jpg
Index: 505, Filename: 20190525_0005_32d3b42dee.jpg, Distance: 0.00000
Index: 546, Filename: 20190525_0005_32d3b42dee_45.jpg, Distance: 0.11169
Index: 507, Filename: 20190525_0005_32d3b42dee_10.jpg, Distance: 0.11401
Index: 528, Filename: 20190525_0005_32d3b42dee_29.jpg, Distance: 0.12564
Index: 535, Filename: 20190525_0005_32d3b42dee_35.jpg, Distance: 0.12807
Feature extraction time: 25.71976 milliseconds
Search time: 0.76917 milliseconds
Query image: ./query/20190525_0006_19969d7a10.jpg
Index: 606, Filename: 20190525_0006_19969d7a10.jpg, Distance: 0.00000
Index: 645, Filename: 20190525_0006_19969d7a10_43.jpg, Distance: 0.10394
Index: 676, Filename: 20190525_0006_19969d7a10_71.jpg, Distance: 0.11919
Index: 672, Filename: 20190525_0006_19969d7a10_68.jpg, Distance: 0.12018
Index: 657, Filename: 20190525_0006_19969d7a10_54.jpg, Distance: 0.12137
Feature extraction time: 22.95796 milliseconds
Search time: 0.55961 milliseconds
Query image: ./query/20190525_0007_b8fc3fd5d6.jpg
Index: 707, Filename: 20190525_0007_b8fc3fd5d6.jpg, Distance: 0.00000
Index: 783, Filename: 20190525_0007_b8fc3fd5d6_77.jpg, Distance: 0.09123
Index: 5050, Filename: 20190525_0050_03c42b6967.jpg, Distance: 0.09190
Index: 711, Filename: 20190525_0007_b8fc3fd5d6_11.jpg, Distance: 0.10301
Index: 5127, Filename: 20190525_0050_03c42b6967_78.jpg, Distance: 0.10349
Feature extraction time: 24.48461 milliseconds
Search time: 0.63475 milliseconds
Query image: ./query/20190525_0008_37270aae38.jpg
Index: 808, Filename: 20190525_0008_37270aae38.jpg, Distance: 0.00000
Index: 904, Filename: 20190525_0008_37270aae38_95.jpg, Distance: 0.08558
Index: 894, Filename: 20190525_0008_37270aae38_86.jpg, Distance: 0.09968
Index: 841, Filename: 20190525_0008_37270aae38_38.jpg, Distance: 0.10724
Index: 831, Filename: 20190525_0008_37270aae38_29.jpg, Distance: 0.10790
Feature extraction time: 24.15350 milliseconds
Search time: 0.58320 milliseconds
Query image: ./query/20190525_0009_7c3de64969.jpg
Index: 909, Filename: 20190525_0009_7c3de64969.jpg, Distance: 0.00000
Index: 975, Filename: 20190525_0009_7c3de64969_68.jpg, Distance: 0.09461
Index: 8888, Filename: 20190525_0088_57f3d2be4b.jpg, Distance: 0.09595
Index: 953, Filename: 20190525_0009_7c3de64969_48.jpg, Distance: 0.11713
Index: 917, Filename: 20190525_0009_7c3de64969_15.jpg, Distance: 0.11743
Feature extraction time: 24.03652 milliseconds
Search time: 0.87667 milliseconds
Query image: ./query/20190525_0082_05b34dfe35.jpg
Index: 8282, Filename: 20190525_0082_05b34dfe35.jpg, Distance: 0.00000
Index: 4848, Filename: 20190525_0048_d3cddd9ef3.jpg, Distance: 0.09383
Index: 2323, Filename: 20190525_0023_4e3a2535c8.jpg, Distance: 0.11490
Index: 2359, Filename: 20190525_0023_4e3a2535c8_40.jpg, Distance: 0.11919
Index: 2525, Filename: 20190525_0025_3acff8485b.jpg, Distance: 0.12424
Feature extraction time: 24.05197 milliseconds
Search time: 0.74041 milliseconds
Query image: ./query/20190525_0083_5897e1c17b.jpg
Index: 8383, Filename: 20190525_0083_5897e1c17b.jpg, Distance: 0.00000
Index: 8468, Filename: 20190525_0083_5897e1c17b_85.jpg, Distance: 0.09814
Index: 8438, Filename: 20190525_0083_5897e1c17b_58.jpg, Distance: 0.10001
Index: 8456, Filename: 20190525_0083_5897e1c17b_74.jpg, Distance: 0.10043
Index: 8448, Filename: 20190525_0083_5897e1c17b_67.jpg, Distance: 0.10328
Feature extraction time: 23.36284 milliseconds
Search time: 0.69641 milliseconds
Query image: ./query/20190525_0084_e104a84e60.jpg
Index: 8484, Filename: 20190525_0084_e104a84e60.jpg, Distance: 0.00000
Index: 8522, Filename: 20190525_0084_e104a84e60_42.jpg, Distance: 0.21986
Index: 8543, Filename: 20190525_0084_e104a84e60_61.jpg, Distance: 0.23164
Index: 8506, Filename: 20190525_0084_e104a84e60_28.jpg, Distance: 0.23545
Index: 8487, Filename: 20190525_0084_e104a84e60_100.jpg, Distance: 0.23896
Feature extraction time: 23.54100 milliseconds
Search time: 0.56206 milliseconds
Query image: ./query/20190525_0085_9c3c6bc7d8.jpg
Index: 8585, Filename: 20190525_0085_9c3c6bc7d8.jpg, Distance: 0.00000
Index: 8620, Filename: 20190525_0085_9c3c6bc7d8_4.jpg, Distance: 0.14355
Index: 8663, Filename: 20190525_0085_9c3c6bc7d8_79.jpg, Distance: 0.16225
Index: 8632, Filename: 20190525_0085_9c3c6bc7d8_50.jpg, Distance: 0.17066
Index: 8595, Filename: 20190525_0085_9c3c6bc7d8_17.jpg, Distance: 0.17368
Feature extraction time: 25.46150 milliseconds
Search time: 0.88545 milliseconds
Query image: ./query/20190525_0086_4cd57b6244.jpg
Index: 8686, Filename: 20190525_0086_4cd57b6244.jpg, Distance: 0.00000
Index: 8762, Filename: 20190525_0086_4cd57b6244_77.jpg, Distance: 0.10554
Index: 8757, Filename: 20190525_0086_4cd57b6244_72.jpg, Distance: 0.12015
Index: 8717, Filename: 20190525_0086_4cd57b6244_36.jpg, Distance: 0.12471
Index: 8722, Filename: 20190525_0086_4cd57b6244_40.jpg, Distance: 0.12904
Feature extraction time: 24.33045 milliseconds
Search time: 0.76031 milliseconds
Query image: ./query/20190525_0087_dc908ca4d5.jpg
Index: 8787, Filename: 20190525_0087_dc908ca4d5.jpg, Distance: 0.00000
Index: 8805, Filename: 20190525_0087_dc908ca4d5_24.jpg, Distance: 0.08822
Index: 8790, Filename: 20190525_0087_dc908ca4d5_100.jpg, Distance: 0.10004
Index: 8808, Filename: 20190525_0087_dc908ca4d5_27.jpg, Distance: 0.11150
Index: 8871, Filename: 20190525_0087_dc908ca4d5_84.jpg, Distance: 0.11718
Feature extraction time: 23.93582 milliseconds
Search time: 0.59247 milliseconds
Query image: ./query/20190525_0088_57f3d2be4b.jpg
Index: 8888, Filename: 20190525_0088_57f3d2be4b.jpg, Distance: 0.00000
Index: 909, Filename: 20190525_0009_7c3de64969.jpg, Distance: 0.09595
Index: 994, Filename: 20190525_0009_7c3de64969_85.jpg, Distance: 0.10479
Index: 917, Filename: 20190525_0009_7c3de64969_15.jpg, Distance: 0.10643
Index: 934, Filename: 20190525_0009_7c3de64969_30.jpg, Distance: 0.12177
Feature extraction time: 24.78324 milliseconds
Search time: 0.59232 milliseconds
Query image: ./query/20190525_0089_a2c93bd374.jpg
Index: 8989, Filename: 20190525_0089_a2c93bd374.jpg, Distance: 0.00000
Index: 3535, Filename: 20190525_0035_67b9426ac9.jpg, Distance: 0.04772
Index: 3620, Filename: 20190525_0035_67b9426ac9_85.jpg, Distance: 0.09340
Index: 9003, Filename: 20190525_0089_a2c93bd374_20.jpg, Distance: 0.10334
Index: 3555, Filename: 20190525_0035_67b9426ac9_26.jpg, Distance: 0.10353
Feature extraction time: 23.07715 milliseconds
Search time: 0.50483 milliseconds
Query image: ./query/20190525_0090_4941737983.jpg
Index: 9090, Filename: 20190525_0090_4941737983.jpg, Distance: 0.00000
Index: 9120, Filename: 20190525_0090_4941737983_35.jpg, Distance: 0.09463
Index: 9186, Filename: 20190525_0090_4941737983_95.jpg, Distance: 0.09667
Index: 9142, Filename: 20190525_0090_4941737983_55.jpg, Distance: 0.12223
Index: 9145, Filename: 20190525_0090_4941737983_58.jpg, Distance: 0.12481
Feature extraction time: 21.57293 milliseconds
Search time: 0.57723 milliseconds
Query image: ./query/20190525_0091_71be77efe3.jpg
Index: 9191, Filename: 20190525_0091_71be77efe3.jpg, Distance: 0.00000
Index: 9286, Filename: 20190525_0091_71be77efe3_94.jpg, Distance: 0.07018
Index: 9198, Filename: 20190525_0091_71be77efe3_14.jpg, Distance: 0.10283
Index: 9261, Filename: 20190525_0091_71be77efe3_71.jpg, Distance: 0.10884
Index: 9209, Filename: 20190525_0091_71be77efe3_24.jpg, Distance: 0.10900
Feature extraction time: 20.78180 milliseconds
Search time: 0.51131 milliseconds
search_time_avg time: 0.65775 milliseconds
```



-----

### 2.4 IndexIVFPQ

>IndexIVFPQ 是 Faiss 中一种常用的索引类型，它使用了一种称为 Product Quantization（PQ）的向量量化技术，将高维特征向量划分为多个子空间，并对每个子空间进行独立的向量量化，从而降低了索引的维度和内存消耗。下面是 IndexIVFPQ 的一些优缺点：
>
>优点：
>
>​	高效的查询速度：由于 IndexIVFPQ 索引将高维特征向量划分为多个子空间，并对每个子空间进行独立的向量量化，因此可以有效地减少查询时间，并在保持较高检索准确率的同时提高查询速度。	
>
>​	适用于大规模数据集：IndexIVFPQ 索引在内存消耗方面比一些其他索引类型，如 IndexFlatL2，更加高效，可以应用于大规模数据集的检索。
>
>​	可以通过调整参数进行优化：IndexIVFPQ 索引的性能可以通过调整一些参数进行优化，如子空间数量、子向量数量等等。通过调整这些参数，可以在准确率与速度之间进行平衡，从而达到最佳的性能。
>
>缺点：
>
>​	索引构建时间较长：相对于一些其他索引类型，如 IndexFlatL2，IndexIVFPQ 索引的构建时间可能会更长，这是因为需要对每个子空间进行向量量化。
>
>​	参数调整较为困难：IndexIVFPQ 索引的性能可以通过调整一些参数进行优化，但这些参数之间的相互作用比较复杂，因此参数调整比较困难。
>
>​	对数据集的质量要求较高：IndexIVFPQ 索引对数据集的质量要求较高，需要保证数据集的分布足够均匀，否则可能会导致索引的准确率下降。
>
>总的来说，IndexIVFPQ 索引是 Faiss 中一种高效的索引类型，适用于大规模数据集的检索，但需要对参数进行适当的调整，并要求数据集的质量较高。

2.4.1 抽特征，得索引

```bash
root@d127cfa55267:/workspace/faiss_test_car# python main_IndexIVFPQ_GPU.py
Extracting features from 9292 images took 157453.86 ms.
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 9292 points to 256 centroids: please provide at least 9984 training points
Adding 9292 feature vectors to index took 2037.56 ms.
Saving index and image names to disk took 30.80 ms.
```

2.4.2 据索引查询

```bash
root@d127cfa55267:/workspace/faiss_test_car# python main_IndexIVFPQ_GPU_query.py
Query image: ./query/20190525_0000_44c5974b76.jpg
Index: 0, Filename: 20190525_0000_44c5974b76.jpg, Distance: 0.09338
Index: 35, Filename: 20190525_0000_44c5974b76_4.jpg, Distance: 0.12409
Index: 65, Filename: 20190525_0000_44c5974b76_67.jpg, Distance: 0.12740
Index: 92, Filename: 20190525_0000_44c5974b76_91.jpg, Distance: 0.12784
Index: 95, Filename: 20190525_0000_44c5974b76_94.jpg, Distance: 0.12854
Feature extraction time: 772.81259 milliseconds
Search time: 1.45787 milliseconds
Query image: ./query/20190525_0001_814ed64fa4.jpg
Index: 101, Filename: 20190525_0001_814ed64fa4.jpg, Distance: 0.04665
Index: 179, Filename: 20190525_0001_814ed64fa4_79.jpg, Distance: 0.10131
Index: 120, Filename: 20190525_0001_814ed64fa4_25.jpg, Distance: 0.10753
Index: 188, Filename: 20190525_0001_814ed64fa4_87.jpg, Distance: 0.12493
Index: 169, Filename: 20190525_0001_814ed64fa4_7.jpg, Distance: 0.12592
Feature extraction time: 25.66230 milliseconds
Search time: 1.10546 milliseconds
Query image: ./query/20190525_0002_587962ae63.jpg
Index: 202, Filename: 20190525_0002_587962ae63.jpg, Distance: 0.08028
Index: 272, Filename: 20190525_0002_587962ae63_71.jpg, Distance: 0.08296
Index: 289, Filename: 20190525_0002_587962ae63_87.jpg, Distance: 0.08397
Index: 276, Filename: 20190525_0002_587962ae63_75.jpg, Distance: 0.08397
Index: 273, Filename: 20190525_0002_587962ae63_72.jpg, Distance: 0.08397
Feature extraction time: 23.03786 milliseconds
Search time: 1.17318 milliseconds
Query image: ./query/20190525_0003_d427e73752.jpg
Index: 303, Filename: 20190525_0003_d427e73752.jpg, Distance: 0.09394
Index: 401, Filename: 20190525_0003_d427e73752_97.jpg, Distance: 0.11445
Index: 336, Filename: 20190525_0003_d427e73752_38.jpg, Distance: 0.11910
Index: 360, Filename: 20190525_0003_d427e73752_6.jpg, Distance: 0.11927
Index: 315, Filename: 20190525_0003_d427e73752_19.jpg, Distance: 0.12335
Feature extraction time: 25.29238 milliseconds
Search time: 1.11702 milliseconds
Query image: ./query/20190525_0004_668fd3b75c.jpg
Index: 404, Filename: 20190525_0004_668fd3b75c.jpg, Distance: 0.09877
Index: 504, Filename: 20190525_0004_668fd3b75c_99.jpg, Distance: 0.12148
Index: 502, Filename: 20190525_0004_668fd3b75c_97.jpg, Distance: 0.12148
Index: 467, Filename: 20190525_0004_668fd3b75c_65.jpg, Distance: 0.12148
Index: 407, Filename: 20190525_0004_668fd3b75c_100.jpg, Distance: 0.12226
Feature extraction time: 25.40604 milliseconds
Search time: 1.27422 milliseconds
Query image: ./query/20190525_0005_32d3b42dee.jpg
Index: 505, Filename: 20190525_0005_32d3b42dee.jpg, Distance: 0.08083
Index: 513, Filename: 20190525_0005_32d3b42dee_15.jpg, Distance: 0.08682
Index: 524, Filename: 20190525_0005_32d3b42dee_25.jpg, Distance: 0.08682
Index: 535, Filename: 20190525_0005_32d3b42dee_35.jpg, Distance: 0.08697
Index: 526, Filename: 20190525_0005_32d3b42dee_27.jpg, Distance: 0.09046
Feature extraction time: 23.51925 milliseconds
Search time: 1.27562 milliseconds
Query image: ./query/20190525_0006_19969d7a10.jpg
Index: 606, Filename: 20190525_0006_19969d7a10.jpg, Distance: 0.09148
Index: 637, Filename: 20190525_0006_19969d7a10_36.jpg, Distance: 0.09798
Index: 695, Filename: 20190525_0006_19969d7a10_89.jpg, Distance: 0.09798
Index: 645, Filename: 20190525_0006_19969d7a10_43.jpg, Distance: 0.09862
Index: 684, Filename: 20190525_0006_19969d7a10_79.jpg, Distance: 0.09862
Feature extraction time: 25.32973 milliseconds
Search time: 1.33516 milliseconds
Query image: ./query/20190525_0007_b8fc3fd5d6.jpg
Index: 707, Filename: 20190525_0007_b8fc3fd5d6.jpg, Distance: 0.05247
Index: 711, Filename: 20190525_0007_b8fc3fd5d6_11.jpg, Distance: 0.06647
Index: 772, Filename: 20190525_0007_b8fc3fd5d6_67.jpg, Distance: 0.06767
Index: 783, Filename: 20190525_0007_b8fc3fd5d6_77.jpg, Distance: 0.06813
Index: 5069, Filename: 20190525_0050_03c42b6967_25.jpg, Distance: 0.07181
Feature extraction time: 24.86823 milliseconds
Search time: 1.13382 milliseconds
Query image: ./query/20190525_0008_37270aae38.jpg
Index: 808, Filename: 20190525_0008_37270aae38.jpg, Distance: 0.06346
Index: 904, Filename: 20190525_0008_37270aae38_95.jpg, Distance: 0.08117
Index: 894, Filename: 20190525_0008_37270aae38_86.jpg, Distance: 0.08474
Index: 831, Filename: 20190525_0008_37270aae38_29.jpg, Distance: 0.08797
Index: 841, Filename: 20190525_0008_37270aae38_38.jpg, Distance: 0.09085
Feature extraction time: 26.36109 milliseconds
Search time: 1.14175 milliseconds
Query image: ./query/20190525_0009_7c3de64969.jpg
Index: 909, Filename: 20190525_0009_7c3de64969.jpg, Distance: 0.09523
Index: 920, Filename: 20190525_0009_7c3de64969_18.jpg, Distance: 0.10487
Index: 994, Filename: 20190525_0009_7c3de64969_85.jpg, Distance: 0.10572
Index: 8888, Filename: 20190525_0088_57f3d2be4b.jpg, Distance: 0.10707
Index: 917, Filename: 20190525_0009_7c3de64969_15.jpg, Distance: 0.11110
Feature extraction time: 24.92217 milliseconds
Search time: 1.00782 milliseconds
Query image: ./query/20190525_0082_05b34dfe35.jpg
Index: 8282, Filename: 20190525_0082_05b34dfe35.jpg, Distance: 0.05975
Index: 2323, Filename: 20190525_0023_4e3a2535c8.jpg, Distance: 0.05975
Index: 4848, Filename: 20190525_0048_d3cddd9ef3.jpg, Distance: 0.06461
Index: 2525, Filename: 20190525_0025_3acff8485b.jpg, Distance: 0.06461
Index: 2572, Filename: 20190525_0025_3acff8485b_50.jpg, Distance: 0.10471
Feature extraction time: 22.80950 milliseconds
Search time: 1.01261 milliseconds
Query image: ./query/20190525_0083_5897e1c17b.jpg
Index: 8383, Filename: 20190525_0083_5897e1c17b.jpg, Distance: 0.07020
Index: 8468, Filename: 20190525_0083_5897e1c17b_85.jpg, Distance: 0.07906
Index: 8457, Filename: 20190525_0083_5897e1c17b_75.jpg, Distance: 0.07922
Index: 8405, Filename: 20190525_0083_5897e1c17b_28.jpg, Distance: 0.07922
Index: 8407, Filename: 20190525_0083_5897e1c17b_3.jpg, Distance: 0.07992
Feature extraction time: 21.71764 milliseconds
Search time: 1.07289 milliseconds
Query image: ./query/20190525_0084_e104a84e60.jpg
Index: 8484, Filename: 20190525_0084_e104a84e60.jpg, Distance: 0.14701
Index: 8506, Filename: 20190525_0084_e104a84e60_28.jpg, Distance: 0.21784
Index: 8543, Filename: 20190525_0084_e104a84e60_61.jpg, Distance: 0.21915
Index: 8504, Filename: 20190525_0084_e104a84e60_26.jpg, Distance: 0.21951
Index: 8512, Filename: 20190525_0084_e104a84e60_33.jpg, Distance: 0.22329
Feature extraction time: 21.76732 milliseconds
Search time: 1.03354 milliseconds
Query image: ./query/20190525_0085_9c3c6bc7d8.jpg
Index: 8585, Filename: 20190525_0085_9c3c6bc7d8.jpg, Distance: 0.11540
Index: 8632, Filename: 20190525_0085_9c3c6bc7d8_50.jpg, Distance: 0.13221
Index: 8680, Filename: 20190525_0085_9c3c6bc7d8_94.jpg, Distance: 0.13433
Index: 8673, Filename: 20190525_0085_9c3c6bc7d8_88.jpg, Distance: 0.13536
Index: 8595, Filename: 20190525_0085_9c3c6bc7d8_17.jpg, Distance: 0.13761
Feature extraction time: 20.42983 milliseconds
Search time: 0.99287 milliseconds
Query image: ./query/20190525_0086_4cd57b6244.jpg
Index: 8686, Filename: 20190525_0086_4cd57b6244.jpg, Distance: 0.09286
Index: 8762, Filename: 20190525_0086_4cd57b6244_77.jpg, Distance: 0.11266
Index: 8722, Filename: 20190525_0086_4cd57b6244_40.jpg, Distance: 0.11290
Index: 8699, Filename: 20190525_0086_4cd57b6244_2.jpg, Distance: 0.11537
Index: 8698, Filename: 20190525_0086_4cd57b6244_19.jpg, Distance: 0.11623
Feature extraction time: 19.49230 milliseconds
Search time: 1.03916 milliseconds
Query image: ./query/20190525_0087_dc908ca4d5.jpg
Index: 8787, Filename: 20190525_0087_dc908ca4d5.jpg, Distance: 0.09034
Index: 8790, Filename: 20190525_0087_dc908ca4d5_100.jpg, Distance: 0.10017
Index: 8805, Filename: 20190525_0087_dc908ca4d5_24.jpg, Distance: 0.10764
Index: 8808, Filename: 20190525_0087_dc908ca4d5_27.jpg, Distance: 0.10879
Index: 8826, Filename: 20190525_0087_dc908ca4d5_43.jpg, Distance: 0.11457
Feature extraction time: 19.15271 milliseconds
Search time: 0.96459 milliseconds
Query image: ./query/20190525_0088_57f3d2be4b.jpg
Index: 8888, Filename: 20190525_0088_57f3d2be4b.jpg, Distance: 0.10164
Index: 8915, Filename: 20190525_0088_57f3d2be4b_32.jpg, Distance: 0.10433
Index: 994, Filename: 20190525_0009_7c3de64969_85.jpg, Distance: 0.10777
Index: 909, Filename: 20190525_0009_7c3de64969.jpg, Distance: 0.10786
Index: 980, Filename: 20190525_0009_7c3de64969_72.jpg, Distance: 0.10981
Feature extraction time: 19.23098 milliseconds
Search time: 0.93826 milliseconds
Query image: ./query/20190525_0089_a2c93bd374.jpg
Index: 8989, Filename: 20190525_0089_a2c93bd374.jpg, Distance: 0.06635
Index: 3535, Filename: 20190525_0035_67b9426ac9.jpg, Distance: 0.06635
Index: 3620, Filename: 20190525_0035_67b9426ac9_85.jpg, Distance: 0.07619
Index: 3547, Filename: 20190525_0035_67b9426ac9_19.jpg, Distance: 0.08322
Index: 3633, Filename: 20190525_0035_67b9426ac9_97.jpg, Distance: 0.08322
Feature extraction time: 20.49748 milliseconds
Search time: 1.04408 milliseconds
Query image: ./query/20190525_0090_4941737983.jpg
Index: 9090, Filename: 20190525_0090_4941737983.jpg, Distance: 0.08859
Index: 9161, Filename: 20190525_0090_4941737983_72.jpg, Distance: 0.10204
Index: 9152, Filename: 20190525_0090_4941737983_64.jpg, Distance: 0.10204
Index: 9148, Filename: 20190525_0090_4941737983_60.jpg, Distance: 0.10204
Index: 9171, Filename: 20190525_0090_4941737983_81.jpg, Distance: 0.10204
Feature extraction time: 19.38764 milliseconds
Search time: 1.18370 milliseconds
Query image: ./query/20190525_0091_71be77efe3.jpg
Index: 9191, Filename: 20190525_0091_71be77efe3.jpg, Distance: 0.05569
Index: 9286, Filename: 20190525_0091_71be77efe3_94.jpg, Distance: 0.06801
Index: 9212, Filename: 20190525_0091_71be77efe3_27.jpg, Distance: 0.08167
Index: 9240, Filename: 20190525_0091_71be77efe3_52.jpg, Distance: 0.08481
Index: 9215, Filename: 20190525_0091_71be77efe3_3.jpg, Distance: 0.08673
Feature extraction time: 19.68903 milliseconds
Search time: 0.84388 milliseconds
search_time_avg time: 1.10738 milliseconds
```



-----

### 2.5 IndexHNSWFlat

>优点：不需要训练，基于图检索的改进方法，检索速度极快，10亿级别秒出检索结果，而且召回率几乎可以媲美Flat，能达到惊人的97%。检索的时间复杂度为loglogn，几乎可以无视候选向量的量级了。并且支持分批导入，极其适合线上任务，毫秒级别 RT。
>
>缺点：构建索引极慢，占用内存极大（是Faiss中最大的，大于原向量占用的内存大小）；添加数据不支持指定数据ID，不支持从索引中删除数据。
>
>参数：HNSWx中的x为构建图时每个点最多连接多少个节点，x越大，构图越复杂，查询越精确，当然构建index时间也就越慢，x取4~64中的任何一个整数。
>
>使用情况：内存充裕，并且有充裕的时间来构建index



## 3. 代码结构

> main_xxx_GPU.py  -- 提取9292张图片特征向量建立faiss索引
>
> main_xxx_GPU_query.py -- 提取待测图片faiss查询                                           (xxx 为 IndexFlatL2、IndexIVFFlat、IndexIVFPQ)
>
> data_enhancement.py -- 数据增强
>
> brute-forceSearch.py -- 暴力检索











