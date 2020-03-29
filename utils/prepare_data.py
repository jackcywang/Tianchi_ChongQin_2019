import json
import os

def remove_bg(annotation):

    bg_imgs = set()
    for anno in annotation['annotations']:
        if anno['category_id'] == 0:
            bg_imgs.add(anno['image_id'])

    print('Find {} images with background annotation'.format(len(bg_imgs)))
    bg_images_only = set()
    for img in bg_imgs:
        num_cats = set()  # 统计每一幅图像中所有标注类别数目
        for anno in annotation['annotations']:
            if anno['image_id'] == img:
                num_cats.add(anno['category_id'])
        if len(num_cats)==1:
            bg_images_only.add(img)
    print('Find {} images with only background annotation'.format(len(bg_images_only)))

    images_deleted = []
    for img in annotation['images']:
        if img['id'] in bg_images_only:
            images_deleted.append(img)

    for img in images_deleted:
        annotation['images'].remove(img)

    print('Remove the images with only background category,done!')
    anno_deletes = []   
    for anno in annotation['annotations']:
        if anno['category_id'] == 0:
            anno_deletes.append(anno)
    print('Find background annotations number:{}'.format(len(anno_deletes)))
    
    for anno in anno_deletes:
        annotation['annotations'].remove(anno)
    print('Remove the background annaotations,done!')
    
    bg_category = {'supercategory': '背景', 'id': 0, 'name': '背景'}
    annotation['categories'].remove(bg_category)

    print('=='*20)

def remove_ignore(annotation,ignore_cats):
    """
    Args:
        annotations: the json file 
        ignore_cats: the ignore categories id eg.[6,7,8]
    """
    ignore_images = []
    annos_deleted = []
    for anno in annotation['annotations']:
        if anno['category_id'] in ignore_cats:
            ignore_images.append(anno['image_id'])
            annos_deleted.append(anno)
    print('Find {} ignore annotations'.format(len(annos_deleted)))
    
    images_deleted = []
    for img in annotation['images']:
        if img['id'] in ignore_images:
            images_deleted.append(img)
    print('Find {} ignore images'.format(len(images_deleted)))
    
    # remove the images
    for img in images_deleted:
        annotation['images'].remove(img)
    print('Remove images done! {} images remain'.format(len(annotation['images'])))
    
    # remove the annotations
    for ann in annos_deleted:
        annotation['annotations'].remove(ann)
    print('Remove annotations done! {} annotations remain'.format(len(annotation['annotations'])))
    
    # remove the categories
    rm_cats = []
    for cat in annotation['categories']:
        if cat['id'] in ignore_cats:
            rm_cats.append(cat)
    for cat in rm_cats:
        annotation['categories'].remove(cat)
    print('Remove categories done! {} categories remain'.format(len(annotation['categories'])))
    print('=='*20)


if __name__ == '__main__':

    round1_train = json.load(open('../dataset/chongqing1_round1_train1_20191223/annotations.json','r'))
    remove_bg(round1_train)
    round2_train = json.load(open('../dataset/chongqing1_round2_train_20200213/annotations.json','r'))
    remove_bg(round2_train)

    ignore_cats = [6,7,8]
    remove_ignore(round1_train,ignore_cats)
    remove_ignore(round2_train,ignore_cats)

    
    with open('../dataset/chongqing1_round1_train1_20191223/fixed_round1.json','w') as f:
        json.dump(round1_train,f)
    
    with open('../dataset/chongqing1_round2_train_20200213/fixed_round2.json','w') as f:
        json.dump(round2_train,f)