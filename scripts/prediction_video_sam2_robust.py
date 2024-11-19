# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import numpy as np
from scipy.ndimage import label

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from clicker import evaluate_sample_sam, evaluate_sample_sam2_video
import gc
import torch

import argparse
import json
import os
from typing import Any, Dict, List


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    # default='/home/david/datasets/2DSOD/testData/DUTS/test_images',
    default='datasets/DUTS/DUTS-TE/images',
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=10,
    help="Path to gt",
)

parser.add_argument(
    "--gtpath",
    type=str,
    default='datasets/DUTS/DUTS-TE/gt',
    help="Path to gt",
)


parser.add_argument(
    "--dataset_name",
    type=str,
    default='DUTS',

    # default='../datasets/DUTS/test_images',
    help="Path to either a single input image or folder of images.",
)


parser.add_argument(
    "--output",
    type=str,
    default='prediction/SAMl/SOD/DUTS',
    # default='/home/david/PycharmProjects/segment-anything-main/sam_output/vit_h',
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)


parser.add_argument(
    '--model_type',
    type=str,
    default='sam-l',
    choices=['sam-l', 'sam2-l'],
)

parser.add_argument(
    '--mode',
    type=str,
    default='bbox',
    choices=['mask', 'bbox', 'point'],
)

parser.add_argument(
    '--mode2',
    type=str,
    default='single',
    choices=['single', 'three', 'five'],
)

# parser.add_argument(
#     "--model_type",
#     type=str,
#     default='vit_h',
#     help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
# )

parser.add_argument(
    "--checkpoint",
    type=str,
    default='model_ck/sam_vit_l_0b3195.pth',
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

def find_connect_area(mask):
    label_mask, num_components = label(mask, structure=np.ones((3, 3)))
    bbox = []
    for connected_label in range(1, num_components + 1):
        component_corrds = np.where(label_mask == connected_label)
        min_x = np.min(component_corrds[1])
        min_y = np.min(component_corrds[0])
        max_x = np.max(component_corrds[1])
        max_y = np.max(component_corrds[0])
        bbox.append(np.array([min_x, min_y, max_x, max_y]))
    return bbox


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs
from prediction_video import save_images

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    if args.model_type == 'sam-l' and args.mode == 'auto':
        sam = sam_model_registry['vit_l'](checkpoint='../model_ck/sam_vit_l_0b3195.pth')
        _ = sam.to(device=args.device)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
        amg_kwargs = get_amg_kwargs(args)
        generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
        # generator2 = SamPredictor(sam)
    elif args.model_type == 'sam-l' and (args.mode == 'bbox' or args.mode == 'point'):
        sam = sam_model_registry['vit_l'](checkpoint='../model_ck/sam_vit_l_0b3195.pth')
        _ = sam.to(device=args.device)
        generator = SamPredictor(sam)
    elif args.model_type == 'sam2-l' and args.mode == 'auto':
        sam2 = build_sam2('sam2_hiera_l.yaml', 'model_ck/sam2_hiera_large.pt', apply_postprocessing=False)
        _ = sam2.to(device=args.device)
        generator = SAM2AutomaticMaskGenerator(model=sam2,
                                               # points_per_side=32,
                                               # points_per_batch=64,
                                               # pred_iou_thresh=0.7,
                                               # stability_score_thresh=0.92,
                                               # stability_score_offset=0.7,
                                               # crop_n_layers=1,
                                               # box_nms_thresh=0.7,
                                               # crop_n_points_downscale_factor=1,
                                               )
    elif args.model_type == 'sam2-l' and (args.mode == 'bbox' or args.mode == 'mask'):
        generator = build_sam2_video_predictor('sam2_hiera_l.yaml', '../model_ck/sam2_hiera_large.pt', device=args.device)
    elif args.model_type == 'sam2-l' and args.mode == 'point':
        generator = build_sam2_video_predictor('sam2_hiera_l.yaml', '../model_ck/sam2_hiera_large.pt', device=args.device)
        sam2 = build_sam2('sam2_hiera_l.yaml', '../model_ck/sam2_hiera_large.pt', apply_postprocessing=False)
        _ = sam2.to(device=args.device)
        generator_first = SAM2ImagePredictor(sam2)


    print(f"Eval on datasets:{args.dataset_name}")
    print("Loading dataset...")
    args.output = args.output + '_' + args.mode + '_' + args.mode2
    if args.dataset_name == 'DAVIS16':
        temp_list = os.listdir(args.input)
        video_list = [os.path.join(args.input, f) for f in temp_list]
        video_gt_list = [os.path.join(args.gtpath, f) for f in temp_list]
    elif args.dataset_name == 'CAD':
        temp_list = os.listdir(args.input)
        temp_list.sort()
        video_list = [os.path.join(args.input, f, 'frames') for f in temp_list]
        video_gt_list = [os.path.join(args.gtpath, f, 'gt') for f in temp_list]
    elif args.dataset_name == 'MoCA':
        temp_list = os.listdir(args.input)
        video_list = [os.path.join(args.input, f, 'Imgs') for f in temp_list]
        video_gt_list = [os.path.join(args.gtpath, f, 'GT') for f in temp_list]
    elif args.dataset_name == 'DAVSOD':
        temp_list = os.listdir(args.input)
        video_list = [os.path.join(args.input, f, 'Imgs') for f in temp_list]
        video_gt_list = [os.path.join(args.gtpath, f, 'GT_object_level') for f in temp_list]
    else:
        temp_list = os.listdir(args.input)
        video_list = [os.path.join(args.input, f) for f in temp_list]
        video_gt_list = [os.path.join(args.gtpath, f) for f in temp_list]

    os.makedirs(args.output, exist_ok=True)

    for video, videogt in zip(video_list, video_gt_list):
        gc.collect()
        torch.cuda.empty_cache()
        targets = [
            f for f in os.listdir(video) if not os.path.isdir(os.path.join(video, f))
        ]
        targets = [os.path.join(video, f) for f in targets]
        targets.sort()

        targetsgt = [
            f for f in os.listdir(videogt) if not os.path.isdir(os.path.join(videogt, f))
        ]
        targetsgt = [os.path.join(videogt, f) for f in targetsgt]
        targetsgt.sort()

        inference_state = generator.init_state(video_path=video)
        generator.reset_state(inference_state)

        if args.mode2 == 'single' and args.mode == 'bbox':
            # single frame as prompt
            tgt = targetsgt[0]
            gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
            _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            gt_binary = gt_binary.astype(bool)
            # bbox of first frame init
            bbox = find_connect_area(gt)
            if len(bbox)!=0:
                var_width, var_height = round(gt.shape[0] * 0.01), round(gt.shape[1] * 0.01)
                var_width, var_height = max(var_width, 1), max(var_height, 1)
                change_width, change_height = round(np.random.random() * 2 * var_width - var_width), round(
                    np.random.random() * 2 * var_height - var_height)
                bboxs = []
                for x in bbox:
                    temp  = np.array([max(0,x[0]+change_width),max(0,x[1]+change_height),min(gt.shape[0],x[2]+change_width),min(gt.shape[1],x[3]+change_height)])
                    bboxs.append(temp)
                # bbox = list(map(lambda x:[max(0,x[0]+change_width),max(0,x[1]+change_height),min(gt.shape[0],x[2]+change_width),min(gt.shape[1],x[3]+gt.shape)],bbox))
                bbox = np.array(bboxs)
            # ann_frame_idx = []
            # ann_obj_id = []
            for i in range(len(bbox)):
                # ann_frame_idx.append(0)
                # ann_obj_id.append(bbox[i])
                _, _, _ = generator.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i,
                    box=bbox[i],
                )
        elif args.mode2 == 'three' and args.mode == 'bbox':
            temp = len(targetsgt) // 3
            for i in range(3):
                tgt = targetsgt[i * temp]
                gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
                _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_binary = gt_binary.astype(bool)
                # bbox of first frame init
                bbox = find_connect_area(gt)
                for j in range(len(bbox)):
                    _, _, _ = generator.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=i * temp,
                        obj_id=j,
                        box=bbox[j],
                    )
        elif args.mode2 == 'five' and args.mode == 'bbox':
            temp = len(targetsgt) // 5
            for i in range(5):
                tgt = targetsgt[i * temp]
                gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
                _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_binary = gt_binary.astype(bool)
                # bbox of first frame init
                bbox = find_connect_area(gt)
                for j in range(len(bbox)):
                    _, _, _ = generator.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=i * temp,
                        obj_id=j,
                        box=bbox[j],
                    )
        elif args.mode2 == 'single' and args.mode == 'mask':
            # single frame as prompt
            tgt = targetsgt[0]
            gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
            _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            # 定义迭代次数
            iterations = 5

            # 定义结构元素（核），这里使用3x3的矩阵
            kernel = np.ones((3, 3), np.uint8)

            # 随机选择腐蚀或膨胀操作，并执行5次迭代
            for _ in range(iterations):
                if random.choice(['erode', 'dilate']) == 'erode':
                    gt_binary = cv2.erode(gt_binary, kernel, iterations=1)
                else:
                    gt_binary = cv2.dilate(gt_binary, kernel, iterations=1)

            gt_binary = gt_binary.astype(bool)
            _, _, _ = generator.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=0,
                mask=gt_binary,
            )
        elif args.mode2 == 'three' and args.mode == 'mask':
            temp = len(targetsgt) // 3
            for i in range(3):
                tgt = targetsgt[i * temp]
                gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
                _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_binary = gt_binary.astype(bool)
                _, _, _ = generator.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=i * temp,
                    obj_id=0,
                    mask=gt_binary,
                )
        elif args.mode2 == 'five' and args.mode == 'mask':
            temp = len(targetsgt) // 5
            for i in range(5):
                tgt = targetsgt[i * temp]
                gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
                _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_binary = gt_binary.astype(bool)
                _, _, _ = generator.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=i * temp,
                    obj_id=0,
                    mask=gt_binary,
                )

        if args.mode2 == 'single' and args.mode == 'point':
            # single frame as prompt
            tgt = targetsgt[0]
            gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
            _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            gt_binary = gt_binary.astype(bool)
            evaluate_sample_sam2_video(gt_mask=gt, predictor=generator, max_iou_thr=0.9,
                                max_clicks=6, inference_state=inference_state, ann_frame_idx=0,robust=True)
        elif args.mode2 == 'three' and args.mode == 'point':
            temp = len(targetsgt) // 3
            for i in range(3):
                tgt = targetsgt[i * temp]
                gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
                _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_binary = gt_binary.astype(bool)
                evaluate_sample_sam2_video(gt_mask=gt, predictor=generator, max_iou_thr=0.9,
                                           max_clicks=6, inference_state=inference_state, ann_frame_idx=i * temp)
        elif args.mode2 == 'five' and args.mode == 'point':
            temp = len(targetsgt) // 5
            for i in range(5):
                tgt = targetsgt[i * temp]
                gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
                _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                gt_binary = gt_binary.astype(bool)
                evaluate_sample_sam2_video(gt_mask=gt, predictor=generator, max_iou_thr=0.9,
                                           max_clicks=6, inference_state=inference_state, ann_frame_idx=i * temp)

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in generator.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for i in range(len(targetsgt)):
            if i>=len(video_segments):continue
            frame_segments = video_segments[i]

            for j in range(len(frame_segments)):
                final_mask = np.zeros_like(gt_binary, dtype=bool)
            final_mask = np.logical_or(final_mask, frame_segments[j][0])
            final_mask = (final_mask.astype(np.uint8)) * 255
            # if final_mask.max()>0:
                # print(final_mask.max())
            save_images(targetsgt[i], None, gt, final_mask, None,args)
        # return
        # prompt_mask = None
        # for t, tgt in zip(targets, targetsgt):
        #     print(f"Processing '{t}'...")
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     image = cv2.imread(t)
        #     if image is None:
        #         print(f"Could not load '{t}' as an image, skipping...")
        #         continue
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        #     # gt_path = args.gtpath + '/' + t.split('/')[-1][0:-3] + 'png'
        #     gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
        #     # original_size = gt.shape[:2]
        #     # gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        #     _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        #     gt_binary = gt_binary.astype(bool)evaluate_sample_sam2_video
        #
        #     if args.mode == 'auto':
        #         # image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        #         if prompt_mask is None:
        #             masks = generator.generate(image)
        #             final_mask = np.zeros_like(gt_binary, dtype=bool)
        #             for mask in masks:
        #                 segmentation = mask['segmentation']
        #                 intersection = np.logical_and(segmentation, gt_binary)
        #                 if np.sum(intersection) / np.sum(segmentation) > 0.9:
        #                     final_mask = np.logical_or(final_mask, segmentation)
        #         else:
        #             generator2.set_image(image)
        #             masks = []
        #             for i in range(len(prompt_mask)):
        #                 mask, _, _ = generator2.predict(box=prompt_mask[i], multimask_output=False)
        #                 masks.append(mask[0])
        #             final_mask = np.zeros_like(gt_binary, dtype=bool)
        #             for mask in masks:
        #                 # intersection = np.logical_and(mask, gt_binary)
        #                 # if np.sum(intersection) / np.sum(mask) > 0.9:
        #                 final_mask = np.logical_or(final_mask, mask)
        #         prompt_mask = final_mask.astype(np.uint8)
        #         prompt_mask = find_connect_area(prompt_mask)
        #         final_mask = (final_mask.astype(np.uint8)) * 255
        #         # final_mask = cv2.resize(final_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        #         if args.dataset_name == 'DAVIS16':
        #             save_path = args.output + '/' + tgt.split('/')[-2] + '/' + tgt.split('/')[-1]
        #             os.makedirs(args.output + '/' + tgt.split('/')[-2], exist_ok=True)
        #         else:
        #             save_path = args.output + '/' + tgt.split('/')[-3] + '/' + tgt.split('/')[-1]
        #             os.makedirs(args.output + '/' + tgt.split('/')[-3], exist_ok=True)
        #         cv2.imwrite(save_path, final_mask)
        #     elif args.mode == 'bbox':
        #         if prompt_mask is None:
        #             bbox = find_connect_area(gt)
        #             generator.set_image(image)
        #             masks = []
        #             for i in range(len(bbox)):
        #                 mask, _, _ = generator.predict(box=bbox[i], multimask_output=False)
        #                 masks.append(mask[0])
        #             final_mask = np.zeros_like(gt_binary, dtype=bool)
        #             for mask in masks:
        #                 # intersection = np.logical_and(mask, gt_binary)
        #                 # if np.sum(intersection) / np.sum(mask) > 0.9:
        #                 final_mask = np.logical_or(final_mask, mask)
        #         else:
        #             generator.set_image(image)
        #             masks = []
        #             for i in range(len(prompt_mask)):
        #                 mask, _, _ = generator.predict(box=prompt_mask[i], multimask_output=False)
        #                 masks.append(mask[0])
        #             final_mask = np.zeros_like(gt_binary, dtype=bool)
        #             for mask in masks:
        #                 # intersection = np.logical_and(mask, gt_binary)
        #                 # if np.sum(intersection) / np.sum(mask) > 0.9:
        #                 final_mask = np.logical_or(final_mask, mask)
        #         prompt_mask = final_mask.astype(np.uint8)
        #         prompt_mask = find_connect_area(prompt_mask)
        #         final_mask = (final_mask.astype(np.uint8)) * 255
        #         if args.dataset_name == 'DAVIS16':
        #             save_path = args.output + '/' + tgt.split('/')[-2] + '/' + tgt.split('/')[-1]
        #             os.makedirs(args.output + '/' + tgt.split('/')[-2], exist_ok=True)
        #         else:
        #             save_path = args.output + '/' + tgt.split('/')[-3] + '/' + tgt.split('/')[-1]
        #             os.makedirs(args.output + '/' + tgt.split('/')[-3], exist_ok=True)
        #         cv2.imwrite(save_path, final_mask)
        #     elif args.mode == 'point':
        #         if prompt_mask is None:
        #             final_mask = evaluate_sample_sam(image=image, gt_mask=gt, predictor=generator, max_iou_thr=0.9,
        #                                          max_clicks=6)
        #         else:
        #             final_mask = evaluate_sample_sam(image=image, gt_mask=gt, predictor=generator, max_iou_thr=0.9, prompt_mask=prompt_mask,
        #                                              max_clicks=6)
        #         prompt_mask = final_mask.astype(np.uint8)
        #         final_mask = (final_mask.astype(np.uint8)) * 255
        #         if args.dataset_name == 'DAVIS16':
        #             save_path = args.output + '/' + tgt.split('/')[-2] + '/' + tgt.split('/')[-1]
        #             os.makedirs(args.output + '/' + tgt.split('/')[-2], exist_ok=True)
        #         else:
        #             save_path = args.output + '/' + tgt.split('/')[-3] + '/' + tgt.split('/')[-1]
        #             os.makedirs(args.output + '/' + tgt.split('/')[-3], exist_ok=True)
        #         cv2.imwrite(save_path, final_mask)
    print("Done!")

import torch,random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)