
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, neighbors


nfeatures = 250
extractor = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)

def features(image):
    """ 检测并描述特征
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = extractor.detectAndCompute(gray_image, None)
    return keypoints, descriptors

def build_vocab(descriptor_lst, n_vocab=200):
    """ 构造词典
    """
    kmeans = cluster.KMeans(n_clusters=n_vocab)
    kmeans.fit_transform(descriptor_lst)
    return kmeans

def build_histogram(descriptors, cluster_alg):
    """ 构造直方图
    """
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    for i in cluster_alg.predict(descriptors):
        histogram[i] += 1
    return histogram

def main():
    path = 'data/dog-breed/test/'
    image_descriptor = []
    descriptors_lst = []
    image_descriptors = []
    for p in list(Path(path).iterdir())[: 10]:
        image = cv2.imread(str(p))
        keypoints, descriptors = features(image)
        descriptors_lst.extend(descriptors)
        image_descriptors.append(descriptors)
    
    cluster_alg = build_vocab(descriptors_lst)
    
    for descriptors in image_descriptors:
        histogram = build_histogram(descriptors, cluster_alg)
        print(histogram)
        
    
if __name__ == '__main__':
    main()