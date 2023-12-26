#!/home/lin/software/miniconda3/envs/mmdet3d/bin/python
#coding=utf-8
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
dataroot = "/home/lin/code/datasets/nuscenes"
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True) # 读取数据集

'''
    ['barrier' 'bicycle' 'bicycle_rack' 'bus' 'car' 
    'construction' 'debris' 'motorcycle' 'pedestrian' 'pushable_pullable' 
    'trafficcone' 'trailer''truck']

    # [障碍物 自行车 自行车支架   公共汽车  汽车  
       建筑   碎片   摩托车      行人     可搬动的障碍 
       交通锥 拖车   卡车]

'''



samples = nusc.sample

categories_all = []
categories = []

# 没个样本都有n个标注
for sample in samples:
    for token in sample['anns']:
        annotation = nusc.get("sample_annotation", token)
        anno_size = annotation['size']
        # print(annotation['category_name'])
        categories_all.append(annotation['category_name'])
        category_name = annotation['category_name'].split('.')[1]
        categories.append(category_name)


counts = {}
for i in categories_all:
    counts[i] = counts.get(i, 0) + 1

print(counts.keys())
print(counts.values())


counts = {}
for i in categories:
    counts[i] = counts.get(i, 0) + 1
# print(counts)
print(counts.values())
print(counts.keys())

print("-----------\n")

sort_count =  {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
print(sort_count.values())
print(sort_count.keys())

print("-----------\n")


categories = np.unique(categories)
categories_all = np.unique(categories_all)





print(len(categories))
print(len(categories_all))




print(counts)
print(categories)
print(categories_all)


print(sort_count.values())
print(sort_count.keys())