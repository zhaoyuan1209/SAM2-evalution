import glob
import os
import numpy as np
import cv2
import tqdm


H,W = 256,256
predict_path = "../scripts/prediction_all"
source_path = "../datasets"

def dice_coefficient(pred, gt):
    # 将图像转换为numpy数组
    pred_arr =pred
    gt_arr = gt

    # 计算交集和并集
    intersection = np.sum(pred_arr * gt_arr)
    union = np.sum(pred_arr) + np.sum(gt_arr)

    # 计算Dice系数
    dice = (2. * intersection) / union if union > 0 else 1.0

    return dice


def compute_dice_scores(gt_paths, pred_paths):
    assert len(gt_paths) == len(pred_paths), "Number of ground truth and prediction images must be the same"

    dice_scores = []

    for gt_path, pred_path in zip(gt_paths, pred_paths):
        # 读取图像
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2GRAY)/255
        pre = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2GRAY)/255
        # 计算Dice系数
        dice = dice_coefficient(pre, gt)
        dice_scores.append(dice)
    return np.mean(dice_scores)
datasets_list = [f"{source_path}/ISBI_2015"]
datasets_list.extend(glob.glob(f"{source_path}brats/*"))
for datasets in datasets_list:#"mvtec3d","mvtec","visa"
    dataset_name = datasets.split(os.path.sep)[-1]
    for  model in ["sam2-l","sam-l"]:
        mode_list = ["bbox","point","auto"] if model=="sam-l" else ["bbox","point","mask"]
        for mode in mode_list:#
            prompt_nums = ["_single","_three","_five"] if model=="sam2-l" else [""]
            for prompt_num  in prompt_nums:

                patients = []
                datasets_second_path ="brats" if "brats"  in dataset_name else "."

                pre_path_list = glob.glob(os.path.join(predict_path,datasets_second_path,dataset_name,model+"_"+mode+prompt_num,"pred_mask","*.png"))
                    #
                    # pre_path_list = glob.glob(
                    #     os.path.join("prediction/brats", dataset_name, model + "_" + mode, "pred_mask", "*.png"))

                for pre_path in pre_path_list:
                    pre_path_list = pre_path.split(os.path.sep)
                    patients.append(pre_path_list[-1].split("===")[0])
                patients = list(set(patients))
                if len(patients)==0:
                    log = 'method_name: {} mode:{} prompt_num:{} dataset: {} Dice: {:.4f} '.format(
                        model, mode, prompt_num, dataset_name, 0.)
                    # raise  Exception("dataset "+log)
                    print(log)
                dices = []
                for patient in patients:
                    pre_paths  =glob.glob(os.path.join(predict_path,datasets_second_path, dataset_name, model + "_" + mode+prompt_num, "pred_mask",f"{patient}*"))
                    gt_paths = list(map(lambda x:os.path.join(datasets,"gt",*x.split(os.path.sep)[-1].split("===")),pre_paths))
                    gt_paths =  list(map(lambda x:x[:-3]+"jpg",gt_paths))
                    dices.append(compute_dice_scores(gt_paths,pre_paths))

                # for pre_path in di:
                # for patient in patients:
                #
                #     file_name=  pre_path.split(os.path.sep)[-1]
                #     gt_path = os.path.join(datasets,"gt",file_name)
                #     gt = cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)
                #     pre = cv2.cvtColor(cv2.imread(pre_path),cv2.COLOR_BGR2GRAY)


                log = 'method_name: {} mode:{} prompt_num:{} dataset: {} Dice: {:.4f} '.format(
                    model,mode, prompt_num, dataset_name, np.mean(dices))
                print(log)

# dataset_root = "../datasets/brats"
