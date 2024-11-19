import glob
import os
import shutil
import cv2
import tqdm
import numpy as np
root = "/home/zhaoyuan/disk2/data/mvtec3d"
save = "../datasets/mvtec3d"
os.makedirs(os.path.join(save,"images"),exist_ok=True)
os.makedirs(os.path.join(save,"gt"),exist_ok=True)
for p in tqdm.tqdm(glob.glob(f"{root}/*/test/*/rgb/*.png")):
    gt_path = p.replace("rgb","gt")
    p_list =p.split(os.path.sep)
    save_name = p_list[-5]+"_"+p_list[-3]+"_"+p_list[-1]
    shutil.copy(p,os.path.join(save,"images",save_name))
    if  os.path.exists(gt_path):
        shutil.copy(gt_path,os.path.join(save,"gt",save_name))
    else:
        H,W,_ = cv2.imread(p).shape
        cv2.imwrite(os.path.join(save,"gt",save_name),np.zeros((H,W)))