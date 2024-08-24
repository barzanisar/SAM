import multiprocessing as mp
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from also_selfsup.superpixel_generation.SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import argparse
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from functools import partial



def compute_sam(cam_token, mask_generator, nusc, sp_root, save_image=False, scene_index=-1):
    if save_image:
        assert scene_index != -1 

    cam = nusc.get("sample_data", cam_token)
    mask_filepath = os.path.join(sp_root, cam["token"] + ".png")
    # check if camera has been processed 
    if os.path.exists(mask_filepath):
        print(f'This {os.path.basename(mask_filepath)} has already been processed')
        return
    image = cv2.imread(os.path.join(nusc.dataroot, cam["filename"]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    segments_sam = np.zeros((image.shape[0], image.shape[1]))
    for id, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        segments_sam[m] = id

    im = Image.fromarray(segments_sam.astype(np.uint8))
    im.save(mask_filepath)

    if save_image:
        # save source image
        image = Image.fromarray(image)
        image_path = mask_filepath = os.path.join(sp_root, cam["token"] + f"_scene{scene_index}.png")
        image.save(image_path)
        


def parse_option():
    parser = argparse.ArgumentParser('SAM', add_help=False)
    parser.add_argument('-v', '--version', help='nuscenes version',
                        default='v1.0-mini')
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data/nuscenes')
    parser.add_argument('-s', '--sp_folder', help='superpixels root', type=str,
                        default='./superpixels/nuscenes/superpixels_sam/') 
    parser.add_argument('-p', '--sam_checkpoint', help='path of pretrained model', type=str,
                        default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--start_idx', default=0, help='the start index of the scene from nuscenes to be processed', type=int)
    parser.add_argument('--end_idx', default=850, help='the end index of the scene from nuscenes to be processed', type=int)
    parser.add_argument('--device_num', default='0', help='which gpu to run on', type=str)
    parser.add_argument("--save_source_images", action="store_true", default=False, help="For each superpixel, save image with the same names as cam[token]")
    parser.add_argument('--points_per_side', default=32, help='SAM: The number of points to be sampled along one side of the image', type=int)
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    # mp.set_start_method('spawn')

    args = parse_option()
    sam_checkpoint = args.sam_checkpoint
    model_type = "vit_h"
    device = 'cuda:' + args.device_num
    start_idx = args.start_idx
    end_idx = args.end_idx 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side)
    print(f'The number of points per side for SAM is: {args.points_per_side}')


    nuscenes_path = args.root_folder
    assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"

    nusc = NuScenes(
        version=args.version, dataroot=f'{nuscenes_path}/{args.version}', verbose=False
    )
    
    # Check if the directory exists
    if not os.path.exists(args.sp_folder):
        os.makedirs(args.sp_folder)
        print(f"SAM Superpixel Directory '{args.sp_folder}' created.")
    else:
        print(f"SAM Superpixel Directory '{args.sp_folder}' already exists.")


    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    num_scenes = len(nusc.scene)
    # with mp.Pool(2) as p:
    for scene_idx in tqdm(range(num_scenes)): #range(start_idx, end_idx)
        # if scene_idx >= num_scenes:
        #     break
        scene = nusc.scene[scene_idx]
        scene_name = scene['name']
        print(f'The current scene is {scene_name}')
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            # func = partial(compute_sam, mask_generator=mask_generator, nusc=nusc, sp_root=args.sp_folder, save_image=args.save_source_images, scene_index=scene_idx)
            # p.map(
            #     func,
            #     [
            #         current_sample["data"][camera_name]
            #         for camera_name in camera_list
            #     ],
            # )

            for camera_name in camera_list:
                compute_sam(current_sample["data"][camera_name], mask_generator=mask_generator, nusc=nusc, sp_root=args.sp_folder, save_image=args.save_source_images, scene_index=scene_idx)

            current_sample_token = current_sample["next"]








