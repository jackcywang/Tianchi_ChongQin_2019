import cv2
import os
import json

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
cat2name={0:'瓶盖破损',1:'瓶盖变形',2:'瓶盖坏边',3:'瓶盖打旋',4:'瓶盖断点',5:'酒液杂质',
            6:'瓶身破损',7:'瓶身气泡',8:'正常喷码',9:'异常喷码'}


anno_path = '../dataset/COCO/Annotations/final.json'
img_path = '../dataset/COCO/Images'
with open(anno_path) as f:
    json_info = json.load(f)
for img_info in json_info['images']:
    file_name = img_info['file_name']
    if file_name == 'imgs_0168606_2_flip_0.jpg':
        img_id = img_info['id']
        pic_dir = os.path.join(img_path, file_name)
        img = cv2.imread(pic_dir)
        for ann_info in json_info['annotations']:
            if img_id == ann_info['image_id']:
                bbox = ann_info['bbox']
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[0]+bbox[2]
                ymax = bbox[1]+bbox[3]
                cat = ann_info['category_id']
        
                cv2.putText(img,str(cat),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
        #cv2.namedWindow("testimg",0)
        cv2.resizeWindow("testimg",400,300)
        cv2.imshow(file_name,img)
        cv2.imwrite('img_flip_0.jpg',img)
        cv2.waitKey(0)
            

        