import os 
import cv2
import argparse
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_json_info(ann_path):
    '''导入json文件
    Args:
        json文件路径
    '''
    with open(ann_path) as f:
        json_data = json.load(f)
    return json_data


def flip_img(src,flip_type):
    '''翻转图像
    Args:
        src:输入图像
        flip_type:翻转类型，1水平翻转，0垂直翻转，-1水平垂直翻转
    return:
        fliped_img:翻转后的图像
    '''
    fliped_img = cv2.flip(src,flip_type)
    return fliped_img

def filp_augment(args):

    flip_type = [1,0]    # 1水平翻转 0:垂直翻转
    name_suffix = 'flip' 
    flip_img_ann_from_dir(args.img_path,args.ann_path,args.aug_img_path,args.aug_json_path,name_suffix,flip_type)

    print("====== data augmentation has been done! ======")


def flip_bbox(bbox,flip_type,img_info):
    '''翻转json中的bbox
    Args:bbox type:list [x,y,w,h]
        flip tyepe: 1,0,-1
    '''
    imgw, imgh = img_info['width'], img_info['height']
    flip = bbox.copy()
    if flip_type == 1:
        flip[0] = imgw - bbox[0] - bbox[2]
    if flip_type == 0:
        flip[1] = imgh - bbox[1] - bbox[3]
    # elif flip_type == -1:
    #     bbox[0] = imgw - bbox[0] - bbox[2]
    #     bbox[1] = imgh - bbox[1] - bbox[3]
    # else:
    #     print('flip type err')
    #     return
    return flip
    
    

def flip_json(i,j,images,annos,json_file,img_info,flip_img_name,flip_type):
    '''翻转json文件
    '''
    temp_info = img_info.copy()
    temp_info['file_name'] = flip_img_name
    temp_info['id'] = len(json_file['images']) + i
    images.append(temp_info)
    for ann in json_file['annotations']:
        if ann['image_id'] == img_info['id']:
            j+=1
            ann_info = ann.copy()
            ann_info['image_id'] = temp_info['id']
            ann_info['id'] = len(json_file['annotations']) + j
            ann_info['bbox'] = flip_bbox(ann['bbox'],flip_type,img_info)
            # if flip_type == 1 and ann_info['category_id'] == 8:
            #     ann_info['category_id'] = 
            annos.append(ann_info)
    
    return images,annos,j

def flip_img_ann_from_dir(img_dir,ann_dir,aug_img_dir,aug_anno_dir,name_suffix,flip_type):

    json_data = get_json_info(ann_dir)
    i = 0
    j = 0
    images = []
    annotations = []
    for img_info in tqdm(json_data['images']):
        img_filename = img_info['file_name']
        img = os.path.join(img_dir,img_filename)
        pic = cv2.imread(img,-1)
        if pic is None:
            continue
        types = get_flip_type(pic)
        for tp in types:
            i = i+1
            flip_img_name = img_filename.split('.')[0] + '_' + name_suffix + '_'+str(tp) +'.'+img_filename.split('.')[-1]
            #fliped_img = flip_img(pic,tp)
            img_list,anno_list,j = flip_json(i,j,images,annotations,json_data,img_info, flip_img_name ,tp)
            #cv2.imwrite(os.path.join(aug_img_dir,flip_img_name),fliped_img)
    new_json = create_json(json_data)
    new_json['images'] += img_list
    new_json['annotations'] += anno_list
    with open(aug_anno_dir,'w') as f:
        json.dump(new_json,f)


def get_flip_type(img):
    '''
    Args:
        img_info
    return:
        对于比赛中瓶盖部分进行水平翻转，返回0
        对于比赛中瓶身部分进行垂直翻转，返回1

    '''

    h,w,c = img.shape
    if h==492 and w==658:
        return [1]
    else:
        return [0,1]

def create_json(dataset):
    mode = {}
    mode["info"] = dataset["info"]
    mode["categories"] = dataset["categories"]
    mode["license"] = dataset["license"]
    mode['images']  = []
    mode['annotations'] = []
    return mode  




def merge_img(args):
    ori_img_path = args.img_path
    des_img_path = args.aug_img_path
    for img in tqdm(os.listdir(ori_img_path)):
        full_path = os.path.join(ori_img_path,img)
        shutil.copy(full_path,des_img_path)
    
    print('merging images done!!!!')


def get_annotations(mode_json,total_json):
    i = 0
    for img_info in mode_json['images']:
        for ann in total_json['annotations']:
            if img_info['id'] == ann['image_id']:
                i +=1
                ann_temp = ann.copy()
                ann_temp['id'] = i
                mode_json['annotations'].append(ann_temp)
    return mode_json

def split_data(args):
    json_info = get_json_info(args.aug_json_path)
    images = json_info['images']
    train_file,test_file = train_test_split(images,test_size=0.15,random_state=2020)
    train_json, val_json = create_json(json_info),create_json(json_info)
    train_json['images'],val_json['images'] = train_file,test_file
    train_json = get_annotations(train_json,json_info)
    val_json = get_annotations(val_json,json_info)
    with open(args.train_json_path,'w') as f:
        json.dump(train_json,f)
    with open(args.val_json_path,'w') as f:
        json.dump(val_json,f)

    print('spliting train val data done!')

def merge_ann(ann1,ann2):
    with open(ann1) as f1:
        json1 = json.load(f1)
    with open(ann2) as f2:
        json2 = json.load(f2)
    json1['images'] += json2['images']
    json1['annotations'] += json2['annotations']
    with open('final_annotations.json','w') as f:
        json.dump(json1,f)
    


def get_args():

    parser = argparse.ArgumentParser(description='script to do augmentation')
    parser.add_argument('--img_path',type=str,default = '../dataset/images')
    parser.add_argument('--ann_path',type=str,default= '../dataset/annotations/merge_annotations.json')
    parser.add_argument('--aug_img_path',type=str,default='../dataset/COCO/Images')
    parser.add_argument('--aug_anno_path',type=str,default= '../dataset/COCO/Annotations')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.aug_img_path):
        os.makedirs(args.aug_img_path)
    if not os.path.exists(args.aug_anno_path):
        os.makedirs(args.aug_anno_path)
    args.aug_json_path = '../dataset/COCO/Annotations/aug_annotations_flip.json'
    args.train_json_path = '../dataset/COCO/Annotations/train.json'
    args.val_json_path = '../dataset/COCO/Annotations/val.json'
    filp_augment(args)
    merge_ann(args.ann_path,args.aug_json_path)
    merge_img(args)
    split_data(args)
    


