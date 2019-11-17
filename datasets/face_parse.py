#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 2019
@author: jingyu
"""
#basic code is taken from https://github.com/Shuvrajit9904/PairedCycleGAN-tf

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import os.path as osp
import re
from PIL import Image


mouth_idx = np.arange(48, 68)
right_eyebrow_idx = np.arange(17, 22)
left_eyebrow_idx = np.arange(22, 27)
right_eye_idx = np.arange(36,42)
left_eye_idx = np.arange(42, 48)
nose_idx = np.arange(27, 35)


FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", mouth_idx),
    ("right_eye_eyebrow", np.append(right_eyebrow_idx, right_eye_idx)),
    ("left_eye_eyebrow", np.append(left_eyebrow_idx, left_eye_idx)),
	("nose", nose_idx),
])

shape_pred = './dataset/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_pred)


def parse_save(image, file,rects, predictor, FACIAL_LANDMARKS_IDXS, output_dir):
    
    p = re.compile('(.*).png')
    out_file_init = p.match(file).group(1)
    
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
     
        for (name, idx_arr) in FACIAL_LANDMARKS_IDXS.items():
    
            clone = image.copy() 
    
            (x,y),radius = cv2.minEnclosingCircle(np.array([shape[idx_arr]]))  
            center = (int(x),int(y))  
            radius = int(radius) + 12   
            
            mask = np.zeros(clone.shape, dtype=np.uint8)  
            mask = cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)
            
            result_array = clone & mask
            y_min = max(0, center[1] - radius)
            x_min = max(0, center[0] - radius)
            result_array = result_array[y_min:center[1] + radius,
                                x_min:center[0] + radius, :]

            out_file_name = output_dir + out_file_init + '_'+ name + '.png'
            if name == 'right_eye_eyebrow':
                cv2.imwrite(out_file_name, cv2.flip( result_array, 1 ))
    
            else:
                cv2.imwrite(out_file_name, result_array)
            # output = face_utils.visualize_facial_landmarks(image, shape)
            # cv2.imshow("Image", output)


input_dir = './dataset/A/test'
output_dir = './dataset/A_parse/test/test'

splits = os.listdir(input_dir)
for sp in splits:
    img_fold = os.path.join(input_dir, sp)
    img_list = os.listdir(img_fold)
    num_imgs = len(img_list)
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    for n in range(num_imgs):
        file = img_list[n]
        path = os.path.join(img_fold, file)
        if os.path.isfile(path):
            image = cv2.imread(path,1)
            # image = np.array(image)
            image = imutils.resize(image, width=512)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            rects = detector(gray, 1)
         
            print(file)
            parse_save(image, file, rects, predictor, FACIAL_LANDMARKS_IDXS, output_dir)




