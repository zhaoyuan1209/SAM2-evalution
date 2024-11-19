import glob
import os
import shutil
import cv2
import tqdm
import nibabel as nib
import  random
import numpy as np
import time
from tensorflow.python.data.experimental.ops.testing import sleep

root = "/home/zhaoyuan/disk2/data/Brats2020/MICCAI_BraTS2020_TrainingData"

save_root = "../datasets/brats/brats_2020"
seg_cls = 255#63 127 255

def nii2pngs(nii_path):
    nii_data = nib.load(nii_path)
    img_data = nii_data.get_fdata()

    # 获取nii文件的基名（不包括扩展名）
    # base_name = os.path.splitext(os.path.splitext(file_name)[0])[0]
    res = []
    # 遍历每个slice并保存为PNG
    for i in range(img_data.shape[2]):
        slice_data = img_data[:, :, i]
        # 归一化处理
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        slice_data = (slice_data * 255).astype(np.uint8)

        res.append(slice_data)
        # 保存为PNG文件
    return res
all_segs = glob.glob(f"{root}/*/*_seg.nii")
all_segs = random.sample(all_segs,k=37)
for seg_cls in ["WT","ET","TC"]:
    for moda in ["flair","t1","t1ce","t2"]:
        save = save_root+"_"+moda+"_"+seg_cls
        paths = []

        for gt_path in tqdm.tqdm(all_segs):
            p = gt_path.replace("seg",moda)
            p_list =p.split(os.path.sep)
            save_name = p_list[-1]
            gts = []
            gts = nii2pngs(gt_path)
            gts  =np.stack(gts)
            if seg_cls=="WT":
                gts=(gts>0)*255
            elif seg_cls=="ET":
                gts=(gts==255)*255
            elif seg_cls=="TC":
                gts[gts==63]=255
                gts=(gts==255)*255
            images = nii2pngs(p)
            images = np.stack(images)
            index  =np.argmax(gts.sum((1,2)),axis = 0)
            # gt_path_list =  glob.glob(os.path.join(gt_path,"Patient-1","*"))
            # for gt_file_path  in gt_path_list:
            #     gts.append(cv2.imread(gt_file_path).sum())
            # index = np.argmax(gts,0)
            # print(len(gts[:index+1][::-1]),len(gts[index:]),index,len(gts))
            for i,(gt,img) in enumerate(zip(gts[:index+1][::-1],images[:index+1][::-1])):
                if i==0:
                    paths.append(os.path.join(save,"gt",save_name+"_former"))
                os.makedirs(os.path.join(save,"gt",save_name+"_former"),exist_ok=True)
                os.makedirs(os.path.join(save,"videos",save_name+"_former"),exist_ok=True)
                cv2.imwrite(os.path.join(save,"gt",save_name+"_former",f"{i}.jpg".zfill(10)),gt)
                cv2.imwrite(os.path.join(save,"videos",save_name+"_former",f"{i}.jpg".zfill(10)),img)
                # time.sleep(0.01)

            for i,(gt,img) in enumerate(zip(gts[index:],images[index:])):
                os.makedirs(os.path.join(save,"gt",save_name+"_latter"),exist_ok=True)
                os.makedirs(os.path.join(save,"videos",save_name+"_latter"),exist_ok=True)
                cv2.imwrite(os.path.join(save,"gt",save_name+"_latter",f"{i}.jpg".zfill(10)),gt)
                cv2.imwrite(os.path.join(save,"videos",save_name+"_latter",f"{i}.jpg".zfill(10)),img)
                # time.sleep(0.01)
        # print(len(set(paths)))
        print(len(os.listdir(os.path.join(save,"videos"))))
    # for i,gt_file_path in enumerate(gts[index:]):
    #     os.makedirs(os.path.join(save,"gt",save_name+"_latter"),exist_ok=True)
    #     os.makedirs(os.path.join(save,"videos",save_name+"_latter"),exist_ok=True)
    #     shutil.copy(gt_file_path,os.path.join(save,"gt",save_name+"_latter",f"{i}.jpg".zfill(10)))
    #     shutil.copy(gt_file_path.replace("annot","").replace("_mask1","_flair_pp"),os.path.join(save,"videos",save_name+"_latter",f"{i}.jpg".zfill(10)))


