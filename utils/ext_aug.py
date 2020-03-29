import os, json
from PIL import Image, ImageDraw
from tqdm import tqdm
base_dir = '../../../data/guangdong/' # path to your data dir


def vflip(dataset, path, name):
    save_dir = 'Images_vflip'
    os.makedirs(save_dir, exist_ok=True)
    for img_info in tqdm(dataset['images']):
        img = Image.open(os.path.join(path, img_info['file_name']))
        img = img.transpose(1)
        img.save(os.path.join(save_dir, img_info['file_name']))

    image_id2wh = {i['id']: [i['width'], i['height']] for i in dataset['images']}

    for anno_info in tqdm(dataset['annotations']):
        w, h = image_id2wh[anno_info['image_id']]
        anno_info['bbox'][1] = h - anno_info['bbox'][1] - anno_info['bbox'][3]

        # for idx, seg in enumerate(anno_info['segmentation'][0]):
        #     if idx % 2 == 1:
        #         anno_info['segmentation'][0][idx] = h - seg

    json.dump(dataset, open('{}_vflip.json'.format(name),'w'))


def rotate_90(dataset, path, name):
    save_dir = '../dataset/COCO/Images_r90'
    os.makedirs(save_dir, exist_ok=True)
    for img_info in tqdm(dataset['images'][:30]):
        img = Image.open(os.path.join(path, img_info['file_name']))
        img = img.transpose(2)
        img.save(os.path.join(save_dir, img_info['file_name']))

    image_id2wh = {i['id']: [i['height'], i['width']] for i in dataset['images']}
    for i in dataset['images']:
        temp = i['height']
        i['height'] = i['width']
        i['width'] = temp

    for anno_info in tqdm(dataset['annotations'][:30]):
        w, h = image_id2wh[anno_info['image_id']]
        anno_info['bbox'] = [w - anno_info['bbox'][3], 
                             h - anno_info['bbox'][0] - anno_info['bbox'][2] , 
                             anno_info['bbox'][3], 
                             anno_info['bbox'][2]]

        # for idx, seg in enumerate(anno_info['segmentation'][0]):
        #     if idx % 2 == 1:
        #         anno_info['segmentation'][0][idx] = h - seg

    json.dump(dataset, open('./dataset/anns/{}_r90.json'.format(name),'w'))

def rotate180(dataset, path, name):
    save_dir = '../dataset/COCO/Images_rotate180'
    os.makedirs(save_dir, exist_ok=True)
    for img_info in tqdm(dataset['images']):
        img = Image.open(os.path.join(path, img_info['file_name']))
        img = img.transpose(3)
        img.save(os.path.join(save_dir, img_info['file_name']))

    image_id2wh = {i['id']: [i['width'], i['height']] for i in dataset['images']}

    for anno_info in dataset['annotations']:
        w,h = image_id2wh[anno_info['image_id']]
        anno_info['bbox'] = [w - anno_info['bbox'][0] - anno_info['bbox'][2], 
                             h - anno_info['bbox'][1] - anno_info['bbox'][3] , 
                             anno_info['bbox'][2], 
                             anno_info['bbox'][3]]
        
        # for idx, seg in enumerate(anno_info['segmentation'][0]):
        #     if idx % 2 == 1:
        #         anno_info['segmentation'][0][idx-1], anno_info['segmentation'][0][idx] = \
        #         w- anno_info['segmentation'][0][idx-1], 
        #         h - anno_info['segmentation'][0][idx]
                
    json.dump(dataset, open('{}_rotate180.json'.format(name), 'w'))

path = '../dataset/COCO/Images/'
name = 'train'
dataset = json.load(open('../dataset/COCO/Annotations/train.json'))
rotate_90(dataset, path, name)
#rotate180(dataset, path, name)

# path = '/dataset/COCO/train2020/'
# name = 'train2'
# dataset = json.load(open('./dataset/annotations.json'))
# vflip(dataset, path, name)
# rotate180(dataset, path, name)