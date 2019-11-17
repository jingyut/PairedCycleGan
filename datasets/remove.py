#Basically remove the images to corresponding folders


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
import shutil

def IsSubString(SubStrList,Str):  
    flag=True 
    for substr in SubStrList: 
        if not(substr in Str): 
            flag=False 
        return flag 
# FileList = []
input_dir_test = './dataset/A_parse/test/'


splits = os.listdir(input_dir_test)

for sp in splits:
# move data to corresponding file
    img_fold_test = os.path.join(input_dir_test, sp)
    img_list = os.listdir(img_fold_test)
    for fn in img_list:
        if IsSubString(['left_eye'],fn):
            fullfilename=os.path.join(img_fold_test,fn) 
            shutil.move(fullfilename,'./dataset/A_parse/eyes/test/left')
        elif IsSubString(['right_eye'],fn):
            fullfilename=os.path.join(img_fold_test,fn) 
            shutil.move(fullfilename,'./dataset/A_parse/eyes/test/right')
        if IsSubString(['mouth'],fn):
            fullfilename=os.path.join(img_fold_test,fn) 
            shutil.move(fullfilename,'./dataset/A_parse/mouth/test')
        elif IsSubString(['nose'],fn):
            fullfilename=os.path.join(img_fold_test,fn) 
            shutil.move(fullfilename,'./dataset/A_parse/nose/test')


# search for complete parsed image
    # for n in range(20):
    #     m = "%02d" % n
    #     SubStrList = [str(m)]
    #     img_fold_B = os.path.join(input_dir_B, sp)
    #     img_list = os.listdir(img_fold_B)
    #     for fn in img_list:
    #         # print (SubStrList)
    #         if IsSubString(SubStrList,fn):
    #             fullfilename=os.path.join(img_fold_B,fn) 
    #             FileList.append(fullfilename)
    #     print (len(FileList))
    #     if len(FileList) != 4:
    #         for files in FileList:
    #             os.remove(files)
    #         FileList = []
    #     else:
    #         FileList = []

# remove images not with size (120,120)
        # img_fold_B = os.path.join(input_dir_B, sp)
        # img_list = os.listdir(img_fold_B)
        # for fn in img_list:
        #     # print (SubStrList)
        #     image_path = osp.join(img_fold_B,fn)
        #     image = Image.open(image_path)
        #     if image.size != (120,120):
        #         os.remove(image_path)

    



        