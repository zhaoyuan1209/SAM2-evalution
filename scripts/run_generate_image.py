import os
datasets = [ ]
save_root = "brats/brats_2020"

# for seg_cls in ["WT","ET","TC"]:
#     for moda in ["flair","t1","t1ce","t2"]:
for seg_cls in ["TC"]:
    for moda in ["t1ce","t2"]:
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



for dataset in datasets: # ["mvtec","visa","mvtec3d"]
    for model in ["sam2-l"]:
        for mode in ["mask","bbox","point"]:
            run = ("CUDA_VISIBLE_DEVICES=1 python prediction_video_sam2.py "
                      f"--mode {mode} --mode2 three --model_type {model} --output prediction/{dataset}/{model} "
                      f"--input ../datasets/{dataset}/videos --gtpath "
                      f"../datasets/{dataset}/gt --dataset_name {dataset} ")
            print(run)
            os.system(run)