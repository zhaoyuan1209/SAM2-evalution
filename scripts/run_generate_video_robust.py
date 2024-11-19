import os
datasets = [ ]
save_root = "brats/brats_2020"

for seg_cls in ["WT","ET","TC"]:
    for moda in ["flair"]:
        save = save_root+"_"+moda+"_"+seg_cls
        datasets.append(save)
# for dataset in datasets: # ["mvtec","visa","mvtec3d"]
#     for model in ["sam-l"]:
#         for mode in ["bbox","point","auto"]:
#             run = ("CUDA_VISIBLE_DEVICES=0 python prediction_video.py "
#                       f"--mode {mode} --model_type {model} --output prediction/{dataset}/{model} "
#                       f"--input ../datasets/{dataset}/videos --gtpath "
#                       f"../datasets/{dataset}/gt --dataset_name {dataset} ")
#             print(run)
#             os.system(run)c


for i in range(5):
    for dataset in datasets: # ["mvtec","visa","mvtec3d"]
        for model in ["sam2-l"]:
            for mode in ["mask"]:
                run = ("CUDA_VISIBLE_DEVICES=0 python prediction_video_sam2_robust.py "
                          f"--mode {mode} --mode2 single --model_type {model} --output prediction/{dataset}/Robust_{i}_{model} --seed {i} "
                          f"--input ../datasets/{dataset}/videos --gtpath "
                          f"../datasets/{dataset}/gt --dataset_name {dataset} ")
                print(run)
                os.system(run)