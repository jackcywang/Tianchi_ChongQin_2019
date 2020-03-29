#encoding:utf/8
import sys 
from mmdet.apis import inference_detector, init_detector 
import json 
import os 
import cv2
import numpy as np 
import argparse 
from tqdm import tqdm 
class MyEncoder(json.JSONEncoder): 
    def default(self, obj): 
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj) 



def bbox_overlaps(boxes,query_boxes):
    """
    Args:
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float
    Return:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N,K),dtype=np.float32)
    for k in range(K):
        box_area =( 
            (query_boxes[k,2]-query_boxes[k,0]+1)*
            (query_boxes[k,3]-query_boxes[k,1]+1)
            )
        for n in range(N):
            iw = (
                # x2 - x1
                min(boxes[n,2],query_boxes[k,2])-
                max(boxes[n,0],query_boxes[k,0])+1
            )
            if iw > 0:
                ih = (
                    min(boxes[n,3],query_boxes[k,3])-
                    max(boxes[n,1],query_boxes[k,1])+1
                )
                if ih > 0:
                    ua = np.float32(
                        (boxes[n,2]-boxes[n,0]+1)*
                        (boxes[n,3]-boxes[n,1]+1)+
                        box_area - iw*ih
                    )
                    overlaps[n,k] = iw*ih / ua
    return overlaps
			
def bgbox_delete(support_det, query_det, thresh=0.5):

    """
    Args:
        support_det:[N, 5] each row is [x1 y1 x2 y2, sore]
        query_det  :[M, 5] each row is [x1 y1 x2 y2, sore]
        thresh : to determine whether the box is stationary or moving 
    """
    # deal with no predictions in an image
    if support_det.shape[0]==0 or query_det.shape[0]==0:
        return query_det

    top_boxes = support_det[:, :4]  # the template used to determine where is the background
    all_boxes = query_det[:, :4]    # the query result backgound bbox to be removed 
    
    #(N, M) ndarray of overlap between support_boxes and query_boxes
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
   
    # print('shape before :',query_det.shape)
    inds_bg = np.unique(np.where(top_to_all_overlaps > thresh)[1])
    inds_rmbg = [index for index in range(query_det.shape[0]) if index not in inds_bg]
    
    boxes_rmbg = query_det[inds_rmbg, :]
    # print('shape after rmbg:',boxes_rmbg.shape)
       
    return boxes_rmbg

#generate result 
def result_from_dir(): 
    index_pinggai = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 10, 8: 12, 9: 13, 10: 11} #这类表  看不懂
    index_pingshen_jiuye = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 10, 8: 12, 9: 13, 10: 11}
    # build the model from a config file and a checkpoint file 
    model_small = init_detector(config2make_json[0], model2make_json[0], device='cuda:0') 
    #model_large = init_detector(config2make_json[1], model2make_json[1], device='cuda:0')
    pics = os.listdir(pic_path)

    jiuye_files = [files for files in pics if len(files.split('_'))==3]
    jiuye_files.sort(key=lambda x: x.split('_')[1],reverse=False)
    
    other_files = [file for file in pics if file not in jiuye_files]
    pinggai_files = []
    pingshen_files = []
    for files in other_files:
        file_path = os.path.join(pic_path,files)
        img = cv2.imread(file_path)
        h,w,c = img.shape
        if h==492 and w==658:
            pinggai_files.append(files)
        else:
            pingshen_files.append(files)

    print('--------pinggai:',len(pinggai_files))
    print('--------pingshen:',len(pingshen_files))
    print('----------jiuye:',len(jiuye_files))
        



    submit = {} 
    images = [] 
    annotations = [] 
    num = 0 
    for im in tqdm(pinggai_files): 
        num += 1 
        img = os.path.join(pic_path,im) 
        result_ = inference_detector(model_small, img)
        images_anno = {} 
        images_anno['file_name'] = im 
        images_anno['id'] = num 
        images.append(images_anno)
        for i ,boxes in enumerate(result_,1):
            if len(boxes): 
                defect_label = index_pinggai[i] 
                for box in boxes: 
                    anno = {} 
                    anno['image_id'] = num 
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]] 
                    anno['bbox'][2] = anno['bbox'][2]-anno['bbox'][0] 
                    anno['bbox'][3] = anno['bbox'][3]-anno['bbox'][1]
                    anno['category_id'] = defect_label
                    anno['score'] = float(box[4]) 
                    annotations.append(anno)
    
    
    print('==== {} ping gai done! ===='.format(num))


    for im in tqdm(pingshen_files): 
        num += 1 
        img = os.path.join(pic_path,im) 
        result_ = inference_detector(model_small, img)
        images_anno = {} 
        images_anno['file_name'] = im 
        images_anno['id'] = num 
        images.append(images_anno)
        for i ,boxes in enumerate(result_,1):
            if len(boxes): 
                defect_label = index_pingshen_jiuye[i] 
                for box in boxes: 
                    anno = {} 
                    anno['image_id'] = num 
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]] 
                    anno['bbox'][2] = anno['bbox'][2]-anno['bbox'][0] 
                    anno['bbox'][3] = anno['bbox'][3]-anno['bbox'][1]
                    anno['category_id'] = defect_label
                    anno['score'] = float(box[4]) 
                    annotations.append(anno)
    
    print('==== {} ping shen done! ===='.format(num))
    
    meta = []
    for img in tqdm(jiuye_files):
        num += 1
        img_path = os.path.join(pic_path,img)
        result = inference_detector(model_small,img_path) #list

        img_anno = {'file_name':img,'id':num}
        images.append(img_anno)

        meta.append(result[9])  # attention to None value

        if len(meta)==5: 
            
            bbox1_rmbg = [bgbox_delete(meta[1],meta[0])]
            meta_rmbg = [bgbox_delete(meta[0],meta[i]) for i in [1,2,3,4]]
            bboxes_rmbg = bbox1_rmbg + meta_rmbg

            for i in range(len(bboxes_rmbg)):
                if len(bboxes_rmbg[i]):
                    for bbox in bboxes_rmbg[i]:
                        annos = {}
                        annos['image_id'] = num-4+i
                        annos['category_id'] = 11
                        annos['bbox'] = [round(float(i),2) for i in bbox[0:4]]
                        annos['bbox'][2] = annos['bbox'][2] - annos['bbox'][0]
                        annos['bbox'][3] = annos['bbox'][3] - annos['bbox'][1]

                        annos['score'] = float(bbox[4])
                        annotations.append(annos)
            meta = []
    
    submit['images'] = images 
    submit['annotations'] = annotations

    print('==== {} jiuye done! ===='.format(num))

    with open(json_out_path, 'w') as fp: 
        json.dump(submit, fp, cls=MyEncoder, indent=4, separators=(',', ': ')) 



if __name__ == "__main__":
    models = ['./model_clean-fd47d884.pth']
    configs = ['./cascade_r101_fpn_context.py']
    parser = argparse.ArgumentParser(description="Generate result") 	
    parser.add_argument("-m", "--model",help="Model path",type=str,default=models)
    parser.add_argument("-c", "--config",help="Config path",type=str,default=configs)
    parser.add_argument("-im", "--im_dir",help="Image path",type=str,default='./tcdata/testA/images/')
    parser.add_argument('-o', "--out",help="Save path", type=str,default='./result.json')
    args = parser.parse_args() 
    model2make_json = args.model 
    config2make_json = args.config
    json_out_path = args.out 
    pic_path = args.im_dir 
    result_from_dir()