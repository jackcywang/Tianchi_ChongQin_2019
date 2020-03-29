import os
import json
import shutil
from tqdm import tqdm

'''
原始数字标签         中文标签               新编号标签        
1	                瓶盖破损	                0
2	                瓶盖变形                    1
3	                瓶盖坏边                    2
4	                瓶盖打旋                    3
5	                瓶盖断点                    4
9	                正常喷码                    8
10	                异常喷码                    9
11	                酒液杂质                    5
12	                瓶身破损                    6
13	                瓶身气泡                    7
预赛中的数据标签去除背景后的类别标号是1-10，复赛数据中，去除了初赛中的6，7，8类，新增11，12，13类
'''


def creat_json():
    meta = {}
    meta['info'] = 'Chongqing_tianchi'
    meta['license'] = []
    meta['categories'] = []
    meta['images'] = []
    meta['annotations'] = []
    return meta

def get_json_info(ann_dirs):
    meta = creat_json()
    images = []
    annotations=[]
    for ann_dir in ann_dirs:
        with open(ann_dir) as f:
            json_info = json.load(f) 
        if os.path.basename(ann_dir) == 'fixed_round1.json':
            for cat in json_info['categories']:
                cat['id'] = cat['id'] -1 #将该json所有标号减一，从0开始，er 标号9，10不变,该类别标签中0,6，7，8已去除
                meta['categories'].append(cat)
            for ann in json_info['annotations']:
                ann['category_id'] = ann['category_id']-1
                annotations.append(ann)
            images += json_info['images']
        if os.path.basename(ann_dir) == 'fixed_round2.json':
            for cat in json_info['categories']:
                if cat['id'] >= 11: #将该json文件中的11-13标签变成6-8，
                    cat['id'] = cat['id'] - 5 -1 #例如先将11减5变成6，6再减一变成5
                    meta['categories'].append(cat)
            for ann in json_info['annotations']:
                ann['category_id'] = ann['category_id']-5-1
                ann['image_id'] += 10000 #由于两个json文件的图片序号都从1开始，避免重复，jdon2图片序号随机加一个大于json1最大图片序号的数字
                annotations.append(ann)
            for img in json_info['images']:
                img['id'] += 10000
                images.append(img)
    return meta,images,annotations
        

def merge_img(img_dir,ann_dir,img_merge_dir):
    with open(ann_dir) as f:
        json_data = json.load(f)
    for imginfo in json_data['images']:
        imgpath = os.path.join(img_dir,imginfo['file_name'])
        print(imgpath)
        shutil.copy(imgpath,img_merge_dir)
    
def merge_ann(ann_dirs,ann_merge_dir):
    meta,images,annotations = get_json_info(ann_dirs)
    img_index = 0
    bbox_index = 0
    for img_info in tqdm(images):
        img_index += 1
        img_temp = img_info.copy()
        img_id = img_info['id']
        img_temp['id'] = img_index
        meta['images'].append(img_temp)
        for ann in annotations:
            ann_temp = ann.copy()
            if ann['image_id'] == img_id:
                bbox_index += 1
                ann_temp['image_id'] = img_index
                ann_temp['id'] = bbox_index
                meta['annotations'].append(ann_temp)
    with open(ann_merge_dir,'w') as f:
        json.dump(meta,f)


if __name__ == '__main__':

    img_paths = ['../dataset/chongqing1_round1_train1_20191223/images',
                '../dataset/chongqing1_round2_train_20200213/images']
    ann_paths = ['../dataset/chongqing1_round1_train1_20191223/fixed_round1.json',
                '../dataset/chongqing1_round2_train_20200213/fixed_round2.json']
    img_merge_path = '../dataset/images'
    ann_merge_path = '../dataset/annotations/merge_annotations_2.json'
    # for i in range(len(img_paths)):
    #     img_path = img_paths[i]
    #     ann_path = ann_paths[i]
    #     merge_img(img_path,ann_path,img_merge_path)
    
    merge_ann(ann_paths,ann_merge_path)
    
         