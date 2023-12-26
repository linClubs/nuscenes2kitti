#!/home/lin/software/miniconda3/envs/mmdet3d/bin/python
#coding=utf-8

import cv2
import numpy as np

colormap = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
