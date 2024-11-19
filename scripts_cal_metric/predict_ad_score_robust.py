import glob
import os
import numpy as np
import cv2
import tqdm
import pandas as pd
from tabulate import  tabulate
import torch
import torch.nn.functional as F
from test_score import calculate_AP,calculate_AUROC,calculate_F1max,calculate_AUPR,calculate_Pro
H,W = 256,256
total_res = {}
for i in range(5):
    for datasets in ["mvtec"]:#"mvtec3d","mvtec","visa""btad"
        for  model in ["sam2-l","sam-l"]:
            for mode in ["point"]:#
                gt_list = glob.glob(f"prediction_all/{datasets}/Robust_{i}_{model}_{mode}/logits/*.npy")
                classes = list(set(map(lambda x:x.split(os.path.sep)[-1].split("_")[0],gt_list)))#[:2]
                results = []
                for cls in classes:
                    gts,pres = [],[]
                    for p in tqdm.tqdm(glob.glob(f"prediction_all/{datasets}/Robust_{i}_{model}_{mode}/logits/{cls}*.npy")):
                        pre = np.load(p)[:1]
                        gt_filename = p.split('/')[-1][:-7]
                        gt_path=os.path.join(f"../datasets/{datasets}/gt/{gt_filename}.png")
                        gt = cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)
                        gt[gt>0]=1
                        # pres.append(cv2.resize(pre,(H,W)))
                        pre = F.interpolate(torch.tensor((pre))[None],(H,W)).numpy()
                        while pre.shape[0]==1:
                             pre = pre[0]
                        pres.append(pre)
                        gts.append(cv2.resize(gt,(H,W)))
                    pres = np.stack(pres)
                    gts = np.stack(gts)
                    I_pres =np.max(pres,axis=(1,2))
                    I_gts = np.max(gts,axis=(1,2))

                    ########## Pixel Level################
                    pres = pres.ravel()
                    gts = gts.ravel()
                    P_AUROC = calculate_AUROC(gts, pres)
                    P_AP = calculate_AP(gts, pres)
                    P_AUPR = calculate_AUPR(gts, pres)
                    Pro = calculate_Pro(gts, pres, shape=(H, W))[0][0]
                    P_F1max = calculate_F1max(gts, pres)
                    print(f"P_AUROC:{P_AUROC},P_AP:{P_AP},P_AUPR:{P_AUPR},P_F1max:{P_F1max},Pro:{Pro}")

                    ########## Instance Level################
                    # I_gt, I_pre = np.random.randint(0, 2, (200)), np.random.randint(0, 2, (200))
                    I_AUROC = calculate_AUROC(I_gts, I_pres)
                    I_AP = calculate_AP(I_gts, I_pres)
                    I_AUPR = calculate_AUPR(I_gts, I_pres)
                    I_F1max = calculate_F1max(I_gts, I_pres)
                    results.append([I_AUROC,I_AP,I_AUPR,I_F1max,P_AUROC,P_AP,P_AUPR,Pro,P_F1max])
                ########## Results################
                columns = "CLASSES,I_AUROC,I_AP,I_AUPR,I_F1max,P_AUROC,P_AP,P_AUPR,Pro,P_F1max".split(",")
                all = np.array(results).mean(0)
                results = [[cls]+r for cls,r in zip(classes,results)]
                results.append(["mean"]+all.tolist())
                pd_data = pd.DataFrame(results, index=None)
                print(f"Seed:{i} dataset:{datasets},model:{model},mode:{mode}")
                print(tabulate(pd_data, headers=columns,tablefmt="pipe"))
                res_dict = total_res.get(f"{model}_{mode}",[])
                res_dict.append(all)
                total_res[f"{model}_{mode}"]=res_dict
for key in total_res.keys():
    data = np.array(total_res[key])[:,[0,1,4,5,7]]
    means = np.round(data.mean(axis=0),4)
    std = np.round(data.std(axis=0),4)
    print(key,[f"{m}±{s}" for m,s in zip(means,std)])

