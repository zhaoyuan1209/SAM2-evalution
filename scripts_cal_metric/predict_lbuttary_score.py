import glob
import os
import numpy as np
import cv2
import tqdm
import pandas as pd
from tabulate import  tabulate
from test_score import calculate_AP,calculate_AUROC,calculate_F1max,calculate_AUPR,calculate_Pro
from metric import cal_neg_num_error,cal_pos_num_error,cal_neg_pos_num_acc,cal_neg_location,cal_pos_location,cal_neg_pos_overhang,cal_neg_num_acc,cal_pos_num_acc


source_path = "/home/zhaoyuan/disk2/data/x_ray_pbd_datasets/test/"
predict_path = "../scripts/prediction_all"
H,W = 256,256
def get_pts_from_xywh(xywh_list):
    centroids = []
    for item in xywh_list:
        x = int(item[0] + item[2] / 2)
        y = int(item[1] + item[3] / 2)
        centroids.append([x, y])
    centroids = np.asarray(centroids, dtype=np.int32).reshape(-1, 2)
    centroids = centroids[np.argsort(centroids[:, 1])]
    return centroids.tolist()
def get_coonected_componet_status(img):
    img = np.array(img, dtype="uint8")
    img = img // 255
    retval, labels_cv, stats, centroids = cv2.connectedComponentsWithStats(
        img, ltype=cv2.CV_32S
    )
    return stats


for datasets in ["lithBattary_neg_location"]:#"mvtec3d","mvtec","visa"
    for  model in ["sam-l","sam2-l"]:
        for mode in ["bbox","point","auto"]:#
            for cls in ["difficult","regular","tough"]:
                neg_num_error, pos_num_error, neg_num_acc, pos_num_acc, neg_pos_num_acc, neg_location_error, pos_location_error, neg_pos_overhang_error = cal_neg_num_error(), cal_pos_num_error(), cal_neg_num_acc(), cal_pos_num_acc(), cal_neg_pos_num_acc(), cal_neg_location(), cal_pos_location(), cal_neg_pos_overhang()
                prediction_neg_location = glob.glob(f"{predict_path}/{datasets}/{model}_{mode}/pred_mask/*{cls}*")
                prediction_pos_location = list(map(lambda x:x.replace("neg_location","pos_location"),prediction_neg_location))
                gt_neg_list =list(map( lambda x:os.path.join(source_path,"neg_location",cls,x.split("=")[-1].replace("png","npy")),prediction_neg_location))
                gt_pos_list =list(map( lambda x:os.path.join(source_path,"pos_location",cls,x.split("=")[-1].replace("png","npy")),prediction_neg_location))

                for neg_pre, pos_pre, neg_gt, pos_gt in zip(prediction_neg_location,prediction_pos_location,gt_neg_list,gt_pos_list):
                    neg_pre = cv2.cvtColor(cv2.imread(neg_pre),cv2.COLOR_BGR2GRAY)
                    pos_pre = cv2.cvtColor(cv2.imread(pos_pre),cv2.COLOR_BGR2GRAY)
                    neg_pre = get_coonected_componet_status(neg_pre)
                    pos_pre = get_coonected_componet_status(pos_pre)
                    neg_pre = get_pts_from_xywh(neg_pre[1:])
                    pos_pre = get_pts_from_xywh(pos_pre[1:])

                    neg_pre = sorted(neg_pre, key=lambda x: x[1])
                    pos_pre = sorted(pos_pre, key=lambda x: x[1])

                    neg_gt = np.load(neg_gt)
                    pos_gt = np.load(pos_gt)
                    neg_num_error.update(neg_pre, neg_gt)
                    pos_num_error.update(pos_pre, pos_gt)
                    neg_num_acc.update(neg_pre, neg_gt)
                    pos_num_acc.update(pos_pre, pos_gt)
                    neg_pos_num_acc.update(neg_pre, neg_gt, pos_pre, pos_gt)
                    if len(neg_pre) == len(neg_gt):
                        neg_location_error.update(neg_pre, neg_gt)
                    if len(pos_pre) == len(pos_gt):
                        pos_location_error.update(pos_pre, pos_gt)
                    if len(neg_pre) == len(neg_gt) and len(pos_pre) == len(pos_gt) and len(pos_pre) + 1 == len(neg_pre):
                        neg_pos_overhang_error.update(neg_pre, neg_gt, pos_pre, pos_gt)
                neg_num_error = neg_num_error.show()
                pos_num_error = pos_num_error.show()
                neg_num_acc = neg_num_acc.show()
                pos_num_acc = pos_num_acc.show()
                neg_pos_num_acc = neg_pos_num_acc.show()
                neg_location_error = neg_location_error.show()
                pos_location_error = pos_location_error.show()
                neg_pos_overhang_error = neg_pos_overhang_error.show()
                # log = 'method_name: {} mode:{} dataset: {} neg_num_MAE: {:.4f} pos_num_MAE: {:.4f} neg_num_Acc: {:.4f} pos_num_Acc: {:.4f} neg_pos_num_Acc: {:.4f} neg_location_MAE: {:.4f} pos_location_MAE: {:.4f} neg_pos_overhang_MAE: {:.4f}'.format(
                #     model,mode, cls, neg_num_error, pos_num_error, neg_num_acc, pos_num_acc, neg_pos_num_acc,
                #     neg_location_error, pos_location_error, neg_pos_overhang_error)

                log = 'method_name: {} mode:{} dataset: {} AN_MAE: {:.4f} CN_MAE: {:.4f} AN_Acc: {:.4f} CN_Acc: {:.4f} PN_Acc: {:.4f} AL_MAE: {:.4f} CL_MAE: {:.4f} OH_MAE: {:.4f}'.format(
                    model,mode, cls, neg_num_error, pos_num_error, neg_num_acc, pos_num_acc, neg_pos_num_acc,
                    neg_location_error, pos_location_error, neg_pos_overhang_error)
                print(log)



            #
            # classes = list(set(map(lambda x:x.split(os.path.sep)[-1].split("_")[0],gt_list)))
            # results = []
            # for cls in classes:
            #     gts,pres = [],[]
            #     for p in tqdm.tqdm(glob.glob(f"prediction/{datasets}/{model}_{mode}/logits/{cls}*.npy")):
            #         pre = np.load(p)
            #         gt_filename = p.split('/')[-1][:-7]
            #         gt_path=os.path.join(f"../datasets/{datasets}/gt/{gt_filename}.png")
            #         gt = cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)
            #         gt[gt>0]=1
            #         pres.append(cv2.resize(pre,(H,W)))
            #         gts.append(cv2.resize(gt,(H,W)))
            #     pres = np.stack(pres)
            #     gts = np.stack(gts)
            #     I_pres =np.max(pres,axis=(1,2))
            #     I_gts = np.max(gts,axis=(1,2))
            #
            #     ########## Pixel Level################
            #     pres = pres.ravel()
            #     gts = gts.ravel()
            #     P_AUROC = calculate_AUROC(gts, pres)
            #     P_AP = calculate_AP(gts, pres)
            #     P_AUPR = calculate_AUPR(gts, pres)
            #     Pro = calculate_Pro(gts, pres, shape=(H, W))[0][0]
            #     P_F1max = calculate_F1max(gts, pres)
            #     print(f"P_AUROC:{P_AUROC},P_AP:{P_AP},P_AUPR:{P_AUPR},P_F1max:{P_F1max},Pro:{Pro}")
            #
            #     ########## Instance Level################
            #     I_gt, I_pre = np.random.randint(0, 2, (200)), np.random.randint(0, 2, (200))
            #     I_AUROC = calculate_AUROC(I_gts, I_pres)
            #     I_AP = calculate_AP(I_gts, I_pres)
            #     I_AUPR = calculate_AUPR(I_gts, I_pres)
            #     I_F1max = calculate_F1max(I_gts, I_pres)
            #     results.append([I_AUROC,I_AP,I_AUPR,I_F1max,P_AUROC,P_AP,P_AUPR,Pro,P_F1max])
            # ########## Results################
            # columns = "CLASSES,I_AUROC,I_AP,I_AUPR,I_F1max,P_AUROC,P_AP,P_AUPR,Pro,P_F1max".split(",")
            # all = np.array(results).mean(0)
            # results = [[cls]+r for cls,r in zip(classes,results)]
            # results.append(["mean"]+all.tolist())
            # pd_data = pd.DataFrame(results, index=None)
            # print(f"dataset:{datasets},model:{model},mode:{mode}")
            # print(tabulate(pd_data, headers=columns,tablefmt="pipe"))