import glob 
import os 
import random
import multiprocessing
import traceback
import json 
import sys 
sys.path.append(".")
import argparse

import cv2
import numpy as np 
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa 
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug.parameters as iap
import imgaug
from tqdm import tqdm

from augmentor.cutom_augmentation import OverLayMultipleImage, imgaug_to_normal

from augmentor.custom_augmentation import (LightFlare, OverLayImage, ParallelLight,
                                 SpotLight, imgaug_to_normal, WarpTexture, RandomLine, BlobAndShadow)

def gen(i):
    num_samples = args.max_combination
    if num_samples >1 :
        if random.uniform(0,1) < 0.5:
            num_samples = 1
        else:
            num_samples = random.randint(1, num_samples)
    all_candidates = random.sample(all_input_sample, num_samples)
    
            
    is_scan = True if random.uniform(0, 1) <= 0.005 else False
    

    augmented_imgs = []
    augmented_annos = []

    all_raw_anno = []
    for anno_path in all_candidates:
        img_path = anno_path.replace('.json', '.png')
        # img_path  = anno_path.replace('.json', '.jpg')
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_anno = json.loads(open(anno_path, 'r').read())

        # Resident card
        # if random.uniform(0, 1) < 0:
        #     if random.uniform(0, 1) < 0.3:
        #         watermark_path = "./data/watermark_resident_card/{}.png".format(random.randint(0, 12))
        #         watermark = Image.open(watermark_path)
        #         w, h = watermark.size
        #         ratio_watermark = random.uniform(1, 3)
        #         watermark = watermark.resize((int(ratio_watermark*w), int(ratio_watermark*h)))
        #     else:
        #         watermark_path = "./data/watermark_resident_card/layout.png"
        #         watermark = Image.open(watermark_path)
            
        #     raw_img = Image.fromarray(raw_img)
        #     if random.uniform(0, 1) < 0.3:
        #         color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
        #     else:
        #         color=(255, 255, 255)
        #     opacity = (0.2, 0.7)
        #     raw_img = utils.overlay_transparent(raw_img, watermark, color=color, ratio=opacity)
        #     raw_img = np.array(raw_img['filled_image'])

        psoi = imgaug.PolygonsOnImage(
            [Polygon(each['polygon']) for each in raw_anno], shape=raw_img.shape)

        # First do geometric transform
        try:
            augmented_img, augmented_anno = type_aug.augment_pipeline_1(
                images=[raw_img], polygons=[psoi])
        except AssertionError:
            return
        augmented_img = augmented_img[0]
        augmented_anno = augmented_anno[0]
        # Then overlay the image into a background
        page_polygon_index = 0
        check_page = False
        for index, each in enumerate(raw_anno):
            if each['key'].lower() == 'page':
                page_polygon_index = index
                check_page = True
        if not check_page:
            print("Sample have no page in label! Please check!")
            return None
        
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2RGBA)
        augmented_anno = imgaug_to_normal(augmented_anno, raw_anno)
        page_polygon = np.array(
            [augmented_anno[page_polygon_index]['polygon']], dtype=np.int32)
        alpha_data = np.zeros(augmented_img.shape[:-1])
        alpha_data = cv2.fillPoly(alpha_data, page_polygon, 1)
        augmented_img[:, :, 3] = alpha_data
        augmented_imgs.append(augmented_img)
        augmented_annos.append(augmented_anno)
        all_raw_anno.append(raw_anno)

    # 70% add background
    if num_samples == 1:
        if random.uniform(0, 1)< 0.8:
            augmented_img, augmented_anno = overlayer.augment_image(
                augmented_imgs, augmented_annos, all_raw_anno, is_scan=is_scan)
        else:
            # augmented_img = type_aug.augment_pipeline_blur.augment_image(augmented_img)
            augmented_anno = augmented_annos
    else:
        augmented_img, augmented_anno = overlayer.augment_image(
                augmented_imgs, augmented_annos, all_raw_anno, is_scan=is_scan)

    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGBA2RGB)
    augmented_img = type_aug.augment_pipeline_2.augment_image(augmented_img)

    img_output_path = os.path.join(
        args.output_folder, '{}_{}.{}'.format(args.name_image, i, args.type_image))
    anno_output_path = os.path.join(
        args.output_folder, '{}_{}.{}'.format(args.name_image, i, "json"))

    if is_scan:
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2GRAY)
    else:
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

    
    cv2.imwrite(img_output_path, augmented_img)

    with open(anno_output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_anno, f, ensure_ascii=False, indent=4)
        

def main(args):
    global all_input_sample
    all_input_sample = glob.glob(os.path.join(args.input_folder, '*.json'))
    os.makedirs(args.output_folder, exist_ok=True)
    global overlayer
    overlayer = OverLayMultipleImage()

    global type_aug
    if args.aug_type:
        type_aug = __import__("augmentor."+args.aug_type, 
                        globals(), locals(), [args.aug_type], 0)
        
    else:
        type_aug = __import__("augmentor.normal", 
                        globals(), locals(), ["normal"], 0)

    pool = multiprocessing.Pool(args.num_workers)
    output = list(tqdm(pool.imap(gen, range(args.num_sample)),
                       total=args.num_sample, desc="Combining"))

    pool.terminate()
    # for i in tqdm(range(args.num_sample)):
    #     gen(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine multiple card into one with many of them')
    parser.add_argument('--input_folder', default='./result/',
                        type=str, required=False, help='folder path to input datasets')
    parser.add_argument('--output_folder', default='./result_data/result_combined/',
                        type=str, required=False, help='folder path to output datasets')
    parser.add_argument('--num_sample', default=10,
                        type=int, required=False, help='number of combined samples')
    parser.add_argument('--max_combination', default=2,
                        type=int, required=False, help='max number of card present in a sample')

    parser.add_argument('--aug_type', type=str, help="Type of augmentation pipeline")
    parser.add_argument('--name_image', required=True, type=str, help="Generated image name")
    parser.add_argument('--type_image', default="png", type=str, help="Type of image")

    parser.add_argument('--num_workers', default=8, type=int,
                        required=False, help='number of core use for multiprocessing')
    global args
    args = parser.parse_args()
    main(args)
