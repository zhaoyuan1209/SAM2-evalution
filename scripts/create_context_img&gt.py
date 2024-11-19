import json
import glob
import os

import cv2
import numpy as np
import nibabel as nib
import tqdm
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
os.makedirs("brats2020_incontext",exist_ok=True)
for seg_cls in ["WT","ET","TC"]:
    for modal in ["flair","t1","t1ce","t2"]:
        pat=16

        img = nii2pngs(f"/home/zhaoyuan/disk2/data/Brats2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0{pat}/BraTS20_Training_0{pat}_{modal}.nii")[100]
        gts = nii2pngs(f"/home/zhaoyuan/disk2/data/Brats2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0{pat}/BraTS20_Training_0{pat}_seg.nii")[100]
        if seg_cls=="WT":
            gts=(gts>0)*255
        elif seg_cls=="ET":
            gts=(gts==255)*255
        elif seg_cls=="TC":
            gts[gts==63]=255
            gts=(gts==255)*255
        for path in glob.glob(f"datasets/brats_incontext/brats_2020_{modal}_{seg_cls}/videos/*{modal}*"):
            gt_path = path.replace("videos","gt")
            cv2.imwrite(f"{path}/000000.jpg",img)
            cv2.imwrite(f"{gt_path}/000000.jpg",gts)
            # cv2.imwrite(f"brats2020_incontext/{modal}_{seg_cls}_{pat}_gt.jpg",gts)