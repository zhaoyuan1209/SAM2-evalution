import glob
import os
import shutil
import cv2
import tqdm
import numpy as np
# root = "/home/zhaoyuan/disk2/data/x_ray_pbd_datasets/test/pos_location"
# save = "../datasets/lithBattary_pos_location"
root = "/home/zhaoyuan/disk2/data/x_ray_pbd_datasets/test/neg_location"
save = "../datasets/lithBattary_neg_location"
radius = 2
os.makedirs(os.path.join(save,"images"),exist_ok=True)
os.makedirs(os.path.join(save,"gt"),exist_ok=True)
for gt_path in tqdm.tqdm(glob.glob(f"{root}/*/*")):
    # gt_path = p.replace("rgb","gt")
    p = gt_path.split(os.path.sep)
    p = glob.glob(os.path.join(os.path.sep.join(p[:-3]),"img",p[-1].replace(".npy",".*")))[0]
    H, W, _ = cv2.imread(p).shape
    center_data = np.load(gt_path)
    gt = np.zeros((H,W))

    # 遍历中心点并画圆
    for center in center_data:
        cv2.circle(gt, center, radius, (255), -1)
    p_list =gt_path.split(os.path.sep)
    save_name = p_list[-3]+"="+p_list[-2]+"="+p_list[-1][:-3]+"png"
    shutil.copy(p,os.path.join(save,"images",save_name))
    cv2.imwrite(os.path.join(save,"gt",save_name),gt)