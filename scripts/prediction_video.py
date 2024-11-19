# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import numpy as np
from scipy.ndimage import label

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from clicker import evaluate_sample_sam
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
    "--seed",
    type=int,
    default=10,
    help="Path to gt",
)
parser.add_argument(
    "--input",
    type=str,
    # default='/home/david/datasets/2DSOD/testData/DUTS/test_images',
    default='datasets/DUTS/DUTS-TE/images',
    help="Path to either a single input image or folder of images.",
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
    default='auto',
    choices=['auto', 'bbox', 'point'],
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


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    if args.model_type == 'sam-l' and args.mode == 'auto':
        sam = sam_model_registry['vit_l'](checkpoint='../model_ck/sam_vit_l_0b3195.pth')
        _ = sam.to(device=args.device)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
        amg_kwargs = get_amg_kwargs(args)
        generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    elif args.model_type == 'sam-l' and (args.mode == 'bbox' or args.mode == 'point'):
        sam = sam_model_registry['vit_l'](checkpoint='../model_ck/sam_vit_l_0b3195.pth')
        _ = sam.to(device=args.device)
        generator = SamPredictor(sam)
    elif args.model_type == 'sam2-l' and args.mode == 'auto':
        sam2 = build_sam2('sam2_hiera_l.yaml', '../model_ck/sam2_hiera_large.pt', apply_postprocessing=False)
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
    elif args.model_type == 'sam2-l' and (args.mode == 'bbox' or args.mode == 'point'):
        sam2 = build_sam2('sam2_hiera_l.yaml', '../model_ck/sam2_hiera_large.pt', apply_postprocessing=False)
        _ = sam2.to(device=args.device)
        generator = SAM2ImagePredictor(sam2)


    print(f"Eval on datasets:{args.dataset_name}")
    print("Loading dataset...")
    args.output = args.output + '_' + args.mode
    if args.dataset_name == 'DAVIS16':
        temp_list = os.listdir(args.input)
        video_list = [os.path.join(args.input, f) for f in temp_list]
        video_gt_list = [os.path.join(args.gtpath, f) for f in temp_list]
    elif args.dataset_name == 'CAD':
        temp_list = os.listdir(args.input)
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

        for t, tgt in zip(targets, targetsgt):
            print(f"Processing '{t}'...")
            gc.collect()
            torch.cuda.empty_cache()
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # gt_path = args.gtpath + '/' + t.split('/')[-1][0:-3] + 'png'
            gt = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
            # original_size = gt.shape[:2]
            # gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            gt_binary = gt_binary.astype(bool)

            if args.mode == 'auto':
                # image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                masks = generator.generate(image)
                final_mask = np.zeros_like(gt_binary, dtype=bool)
                for mask in masks:
                    segmentation = mask['segmentation']
                    intersection = np.logical_and(segmentation, gt_binary)
                    if np.sum(intersection) / np.sum(segmentation) > 0.9:
                        final_mask = np.logical_or(final_mask, segmentation)
                final_mask = (final_mask.astype(np.uint8)) * 255
                # final_mask = cv2.resize(final_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                save_images(t, image, gt, final_mask, None,args)
                # if args.dataset_name == 'DAVIS16':
                #     save_path = args.output + '/' + tgt.split('/')[-2] + '/' + tgt.split('/')[-1]
                #     os.makedirs(args.output + '/' + tgt.split('/')[-2], exist_ok=True)
                # else:
                #     save_path = args.output + '/' + tgt.split('/')[-3] + '/' + tgt.split('/')[-1]
                #     os.makedirs(args.output + '/' + tgt.split('/')[-3], exist_ok=True)
                # cv2.imwrite(save_path, final_mask)
            elif args.mode == 'bbox':
                bbox = find_connect_area(gt)
                generator.set_image(image)
                masks = []
                for i in range(len(bbox)):
                    mask, _, _ = generator.predict(box=bbox[i], multimask_output=False)
                    masks.append(mask[0])
                final_mask = np.zeros_like(gt_binary, dtype=bool)
                for mask in masks:
                    # intersection = np.logical_and(mask, gt_binary)
                    # if np.sum(intersection) / np.sum(mask) > 0.9:
                    final_mask = np.logical_or(final_mask, mask)
                final_mask = (final_mask.astype(np.uint8)) * 255

                save_images(t, image, gt, final_mask, None,args)
            elif args.mode == 'point':
                final_mask,_ = evaluate_sample_sam(image=image, gt_mask=gt, predictor=generator, max_iou_thr=0.9,
                                                 max_clicks=6)
                final_mask = (final_mask.astype(np.uint8)) * 255
                save_images(t, image, gt, final_mask, None,args)
        # return
    print("Done!")

def save_images(t,image,gt, final_mask,final_prob,args):
    os.makedirs(args.output + '/show_img/', exist_ok=True)
    os.makedirs(args.output + '/pred_mask/', exist_ok=True)
    os.makedirs(args.output + '/logits/', exist_ok=True)
    final_mask = ((final_mask>0).astype(np.uint8)) * 255
    # plt.subplot(2, 2, 1)
    # plt.imshow(image)
    # plt.subplot(2, 2, 2)
    # plt.imshow(gt)
    # plt.subplot(2, 2, 3)
    # plt.imshow(final_mask)
    # plt.subplot(2, 2, 4)
    # plt.imshow(final_prob)
    # plt.savefig(args.output + '/show_img/' + t.split('/')[-1][0:-3] + 'jpg')
    # plt.clf()
    if final_mask is not None:
        save_path = args.output + '/pred_mask/' + t.split('/')[-2]+"===" +t.split('/')[-1][0:-3] + 'png'
        cv2.imwrite(save_path, final_mask)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)