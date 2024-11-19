import os
datasets = [ ]
# save_root = "brats/brats_2020"
# for dataset in ["mvtec"]:
#     for model in ["sam2-l"]:
#         for mode in ["bbox","point"]:
#             run = ("CUDA_VISIBLE_DEVICES=0 python prediction_img.py "
#                       f"--mode {mode}  --model_type {model} --output prediction/{dataset}/{model} "
#                       f"--input ../datasets/{dataset}/images --gtpath "
#                       f"../datasets/{dataset}/gt --dataset_name {dataset} ")
#             print(run)
#             os.system(run)


for i in range(5):
    for dataset in ["mvtec"]:
        for model in ["sam-l","sam2-l"]:
            for mode in ["bbox"]:#"bbox",
                run = ("CUDA_VISIBLE_DEVICES=0 python prediction_img_robust.py "
                          f"--mode {mode}  --model_type {model} --output prediction/{dataset}/Robust_{i}_{model} "
                          f"--input ../datasets/{dataset}/images  --seed {i} "
                          f"--gtpath  ../datasets/{dataset}/gt --dataset_name {dataset} ")
                print(run)
                os.system(run)