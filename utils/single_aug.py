import os 
import cv2
import argparse
import json
import shutil
from tqdm import tqdm


def get_json_info(ann_path):
    '''导入json文件
    Args:
        json文件路径
    '''
    with open(ann_path) as f:
        json_data = json.load(f)
    return json_data

def create_json(dataset):
    mode = {}
    mode["info"] = dataset["info"]
    mode["categories"] = []
    mode["license"] = dataset["license"]
    mode['images']  = dataset['images']
    mode['annotations'] = []
    return mode  

def split_from_aug(aug_path,round1_ann_path,round2_ann_path):
    json_info = get_json_info(aug_path)
    cat_list = [0,1,2,3,4,8,9]
    round1_json = create_json(json_info)
    round2_json = create_json(json_info)
    for cat_info in json_info['categories']:
        cat_temp = cat_info.copy()
        if cat_info['id'] not in cat_list:
            cat_temp['id'] = cat_temp['id'] - 5
            round2_json['categories'].append(cat_temp)
        else:
            if cat_info['id'] <= 4:
                round1_json['categories'].append(cat_temp)
            if cat_info['id'] >=8:
                cat_temp['id'] = cat_temp['id'] - 3
                round1_json['categories'].append(cat_temp)
        
    num1 = 0
    num2 = 0
    for ann in json_info['annotations']:
        cat_index = ann['category_id']
        ann_temp = ann.copy()
        if cat_index in cat_list:
            num1 += 1
            ann_temp = ann.copy()
            ann_temp['id'] = num1
            if cat_index <=4:
                round1_json['annotations'].append(ann_temp)
            if cat_index >=8:
                ann_temp['category_id'] -= 3
                round1_json['annotations'].append(ann_temp)
        else:
            num2 += 1
            ann_temp['id'] = num2
            ann_temp['category_id'] -= 5
            round2_json['annotations'].append(ann_temp)



    with open(round1_ann_path,'w') as f:
        json.dump(round1_json,f)
    with open(round2_ann_path,'w') as f:
        json.dump(round2_json,f)
    
    
                

        




    


if __name__ == '__main__':
    aug_path = '../dataset/COCO/Annotations/final_annotations.json'
    round1_aug_path = '../dataset/COCO/Annotations/round1_aug.json'
    round2_aug_path = '../dataset/COCO/Annotations/round2_aug.json'
    split_from_aug(aug_path,round1_aug_path,round2_aug_path)
    
    


