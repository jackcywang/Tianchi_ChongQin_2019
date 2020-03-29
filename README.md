# 数智重庆.全球产业赋能创新大赛代码

这是我参加华为云比赛后又参加的一个比赛，比赛连接[link](https://tianchi.aliyun.com/competition/entrance/231763/introduction)
本次比赛代码是基于商汤开源的目标检测框架mmdetection来完成的，这个框架整合了许多最新的目标检测模型，感觉还是很不错的。关于这个mmdetection框架和使用方法,我后续会在自己博客网站中一一介绍。
### 1 代码结构
|-mmdetection (mmdetection框架,部分代码有改动)  
&ensp;  |-congfig  
&ensp; |-dataset  
&ensp;&ensp;&ensp;|chongqing1_round1_train1_20191223  
&ensp;&ensp;&ensp;&ensp;images  
&ensp;&ensp;&ensp;&ensp;annotations.json  
&ensp;&ensp;&ensp;|chongqing1_round2_train_20200213  
&ensp;&ensp;&ensp;&ensp;images  
&ensp;&ensp;&ensp;&ensp;annotations.json  
&ensp;.....  
|-utils(数据清洗，增强，合并)  
|-work_dirs(权重保存)  
submit_v1.py(初赛提交脚本)  
submit_v2.py(复赛提交脚本)  

### 2.Requirements
* Linux
* OS: Ubuntu 16.04
* CUDA 9.0 or higher
* Python 3.5+ 
* PyTorch 1.1 or higher
* mmcv
* tqdm 4.42.1
* NCCL 2
* GCC(G++) 4.9 or higher
* NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
* GCC(G++): 4.9/5.3/5.4/7.3

### Install
可以按照mmdetection的安装指示[install](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md)进行，在安装cocoapi的时候会一直卡着不动，只能重新安装。

#### cocoapi install
```
git clone https://github.com/pdollar/coco.git

cd coco/PythonAPI

python3 setup.py build_ext --inplace

python3 setup.py build_ext install
```
#### Data prepare
数据存放格式如下
|-dataset 
&ensp;&ensp;&ensp;|chongqing1_round1_train1_20191223  
&ensp;&ensp;&ensp;&ensp;images  
&ensp;&ensp;&ensp;&ensp;annotations.json  
&ensp;&ensp;&ensp;|chongqing1_round2_train_20200213  
&ensp;&ensp;&ensp;&ensp;images  

1.清洗数据，删除背景数据
```
cd utils/
python3 prepare_data.py
```
2.合并初赛复赛annotation.json
```
python3 merge_data.py
```
3.数据扩充并划分训练验证集
```
python3 img_aug.py
```

### train
```
cd mmdetecion
python3 tools/train configs/cascade_r101_fpn_context.py
```
### submit 
```
python3 submit_v2.py
```
### mmdetecion代码改动位置
1.mmdetection/mmdet/datasets/coco.py  
2.mmdetection/mmdet/models/roi_extractors/single_level.py  
