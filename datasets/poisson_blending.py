'''python implementation of poisson image blending
'''

# Basic Code is taken from https://github.com/yskmt/pb



import numpy as np
from skimage import data, io
import scipy.sparse
from scipy.sparse import coo_matrix

import matplotlib.pyplot as plt
from PIL import Image
import pdb
import cv2
import os
import imutils
import dlib
import os
import os.path as osp
import re
from imutils import face_utils
from collections import OrderedDict


def create_mask(img_mask, img_target, img_src, offset=(0, 0)):
    '''
    Takes the np.array from the grayscale image
    '''

    # crop img_mask and img_src to fit to the img_target
    hm, wm = img_mask.shape
    ht, wt, nl = img_target.shape

    hd0 = max(0, -offset[0])
    wd0 = max(0, -offset[1])

    hd1 = hm - max(hm + offset[0] - ht, 0)
    wd1 = wm - max(wm + offset[1] - wt, 0)

    mask = np.zeros((hm, wm))
    mask[img_mask > 0] = 1
    mask[img_mask == 0] = 0

    mask = mask[hd0:hd1, wd0:wd1]
    src = img_src[hd0:hd1, wd0:wd1]

    # fix offset
    offset_adj = (max(offset[0], 0), max(offset[1], 0))

    # remove edge from the mask so that we don't have to check the
    # edge condition
    mask[:, -1] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[0, :] = 0

    return mask, src, offset_adj


def get_gradient_sum(img, i, j, h, w):
    """
    Return the sum of the gradient of the source imgae.
    * 3D array for RGB
    """

    v_sum = np.array([0.0, 0.0, 0.0])
    v_sum = img[i, j] * 4 \
        - img[i + 1, j] - img[i - 1, j] - img[i, j + 1] - img[i, j - 1]

    return v_sum


def get_mixed_gradient_sum(img_src, img_target, i, j, h, w, ofs,
                           c=1.0):
    """
    Return the sum of the gradient of the source imgae.
    * 3D array for RGB
    c(>=0): larger, the more important the target image gradient is
    """

    v_sum = np.array([0.0, 0.0, 0.0])
    nb = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for kk in range(4):

        fp = img_src[i, j] - img_src[i + nb[kk, 0], j + nb[kk, 1]]
        gp = img_target[i + ofs[0], j + ofs[1]] \
            - img_target[i + nb[kk, 0] + ofs[0], j + nb[kk, 1] + ofs[1]]

        # if np.linalg.norm(fp) > np.linalg.norm(gp):
        #     v_sum += fp
        # else:
        #     v_sum += gp

        v_sum += np.array([fp[0] if abs(fp[0] * c) > abs(gp[0]) else gp[0],
                           fp[1] if abs(fp[1] * c) > abs(gp[1]) else gp[1],
                           fp[2] if abs(fp[2] * c) > abs(gp[2]) else gp[2]])

    return v_sum


def poisson_blend(img_mask, img_src, img_target, method='mix', c=1.0,
                  offset_adj=(0,0)):

    hm, wm = img_mask.shape
    region_size = hm * wm

    F = np.zeros((region_size, 3))
    A = scipy.sparse.identity(region_size, format='lil')

    get_k = lambda i, j: i + j * hm

    # plane insertion
    if method in ['target', 'src']:
        for i in range(hm):
            for j in range(wm):
                k = get_k(i, j)

                # ignore the edge case (# of neighboor is always 4)
                if img_mask[i, j] == 1:

                    if method == 'target':
                        F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]
                    elif method == 'src':
                        F[k] = img_src[i, j]
                else:
                    F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]

    # poisson blending
    else:
        if method == 'mix':
            grad_func = lambda ii, jj: get_mixed_gradient_sum(
                img_src, img_target, ii, jj, hm, wm, offset_adj, c=c)
        else:
            grad_func = lambda ii, jj: get_gradient_sum(
                img_src, ii, jj, hm, wm)

        for i in range(hm):
            for j in range(wm):
                k = get_k(i, j)

                # ignore the edge case (# of neighboor is always 4)
                if img_mask[i, j] == 1:
                    f_star = np.array([0.0, 0.0, 0.0])

                    if img_mask[i - 1, j] == 1:
                        A[k, k - 1] = -1
                    else:
                        f_star += img_target[i - 1 +
                                             offset_adj[0], j + offset_adj[1]]

                    if img_mask[i + 1, j] == 1:
                        A[k, k + 1] = -1
                    else:
                        f_star += img_target[i + 1 +
                                             offset_adj[0], j + offset_adj[1]]

                    if img_mask[i, j - 1] == 1:
                        A[k, k - hm] = -1
                    else:
                        f_star += img_target[i +
                                             offset_adj[0], j - 1 + offset_adj[1]]

                    if img_mask[i, j + 1] == 1:
                        A[k, k + hm] = -1
                    else:
                        f_star += img_target[i +
                                             offset_adj[0], j + 1 + offset_adj[1]]

                    A[k, k] = 4
                    F[k] = grad_func(i, j) + f_star

                else:
                    F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]

    A = A.tocsr()

    img_pro = np.empty_like(img_target.astype(np.uint8))
    img_pro[:] = img_target.astype(np.uint8)

    for l in range(3):
        # x = pyamg.solve(A, F[:, l], verb=True, tol=1e-15, maxiter=100)
        x = scipy.sparse.linalg.spsolve(A, F[:, l])
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_pro.dtype)

        img_pro[offset_adj[0]:offset_adj[0] + hm,
                offset_adj[1]:offset_adj[1] + wm, l]\
            = x.reshape(hm, wm, order='F')

    return img_pro



### create mask images

# mouth_idx = np.arange(48, 68)
# right_eyebrow_idx = np.arange(17, 22)
# left_eyebrow_idx = np.arange(22, 27)
# right_eye_idx = np.arange(36,42)
# left_eye_idx = np.arange(42, 48)
# nose_idx = np.arange(27, 35)


# FACIAL_LANDMARKS_IDXS = OrderedDict([
#     ("mouth", mouth_idx),
#     ("right_eye_eyebrow", np.append(right_eyebrow_idx, right_eye_idx)),
#     ("left_eye_eyebrow", np.append(left_eyebrow_idx, left_eye_idx)),
# ])

# shape_pred = './datasets/dataset/shape_predictor_68_face_landmarks.dat'

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shape_pred)
# def parse_save(image, file,rects, predictor, FACIAL_LANDMARKS_IDXS, output_dir):
    
#     p = re.compile('(.*).png')
#     out_file_init = p.match(file).group(1)
    
#     for (i, rect) in enumerate(rects):

#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
     
#         for (name, idx_arr) in FACIAL_LANDMARKS_IDXS.items():
    
#             clone = image.copy() 
    
#             (x,y),radius = cv2.minEnclosingCircle(np.array([shape[idx_arr]]))  
#             center = (int(x),int(y))  
#             radius = int(radius) + 12   
            
#             mask = np.zeros(clone.shape, dtype=np.uint8)  
#             mask = cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)
            
#             # result_array = clone & mask
#             # y_min = max(0, center[1] - radius)
#             # x_min = max(0, center[0] - radius)
#             # result_array = result_array[y_min:center[1] + radius,
#             #                     x_min:center[0] + radius, :]

#             out_file_name = output_dir + out_file_init + '_'+ name + '.png'
           
#             cv2.imwrite(out_file_name, mask)

# input_dir = './datasets/dataset/B/test'
# output_dir = './datasets/dataset/B_mask/test/test'
# # def face_parse(input_dir, output_dir):
# splits = os.listdir(input_dir)
# for sp in splits:
#     img_fold = os.path.join(input_dir, sp)
#     img_list = os.listdir(img_fold)
#     num_imgs = len(img_list)
#     print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
#     for n in range(num_imgs):
#         file = img_list[n]
#         path = os.path.join(img_fold, file)
#         if os.path.isfile(path):
#             image = cv2.imread(path,1)
#             # image = np.array(image)
#             image = imutils.resize(image, width=512)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#             rects = detector(gray, 1)
         
#             print(file)
#                 # break
#             parse_save(image, file, rects, predictor, FACIAL_LANDMARKS_IDXS, output_dir)

def pb(path_src):

    offset = (0,0)

    # # split AB image into A and B
    # AB = Image.open(path_src).convert('RGB')
    
    # w, h = AB.size
    # w2 = int(w / 2)
    # A = AB.crop((0, 0, w2, h))
    # B = AB.crop((w2, 0, w, h))
    # path_A_split = './datasets/dataset/AB_split/A.png'
    # path_B_split = './datasets/dataset/AB_split/B.png'
    # A.save(path_A_split, 'PNG')
    # B.save(path_B_split, 'PNG')
    # #
    path_src = './datasets/dataset/B/test/test/001.png'
    path_target = './datasets/dataset/A/test/test/004.png'
    path_mask = './datasets/dataset/B_mask/test/test001_left_eye_eyebrow.png'
    img_src = io.imread(path_src).astype(np.float64)
    img_src = imutils.resize(img_src.copy(), width=512)
    # img_src = img_src.copy()
    # img_src = imutils.resize(img_src,width = 64) 
    img_target = io.imread(path_target)
    img_target = imutils.resize(img_target.copy(), width=512)  
    img_mask = io.imread(path_mask, as_gray=True)
    img_mask, img_src, offset_adj \
        = create_mask(img_mask.astype(np.float64),
                      img_target, img_src, offset=offset)

    img_pro = poisson_blend(img_mask, img_src, img_target,
                            method='normal', offset_adj=offset_adj)
    path_mask = './datasets/dataset/B_mask/test/test001_right_eye_eyebrow.png'
    img_target = img_pro
    img_mask = io.imread(path_mask, as_gray=True)
    img_mask, img_src, offset_adj \
        = create_mask(img_mask.astype(np.float64),
                      img_target, img_src, offset=offset)

    img_pro = poisson_blend(img_mask, img_src, img_target,
                            method='normal', offset_adj=offset_adj)
    path_mask = './datasets/dataset/B_mask/test/test001_mouth.png'
    img_target = img_pro
    img_mask = io.imread(path_mask, as_gray=True)
    img_mask, img_src, offset_adj \
        = create_mask(img_mask.astype(np.float64),
                      img_target, img_src, offset=offset)

    img_pro = poisson_blend(img_mask, img_src, img_target,
                            method='normal', offset_adj=offset_adj)

    plt.imshow(img_pro)
    plt.show()
    path_pro = './datasets/dataset/fake_B_warp/A_01.png'
    io.imsave(path_pro, img_pro)    
  
pb('./datasets/dataset/B/test/test/001.png')