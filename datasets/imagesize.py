from PIL import Image
import os, os.path as OP
print("current dir:\t", OP.abspath(os.getcwd()))
print("app dir:\t", OP.abspath(OP.dirname(__file__)))

def imagesize(input_dir_B):
	splits = os.listdir(input_dir_B)

	for sp in splits:
	    img_fold_B = os.path.join(input_dir_B, sp)
	    img_list = os.listdir(img_fold_B)
	    num_imgs = len(img_list)
	    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
	    for n in range(num_imgs):
	    	file = img_list[n]
	    	path_A = os.path.join(img_fold_B, file)
	    	img = Image.open(path_A)
	    	print(img.size)
# imagesize('./dataset/B_parse/test')
# imagesize('./dataset/B_parse/val')
imagesize('./dataset/AB_parse/eyes')
