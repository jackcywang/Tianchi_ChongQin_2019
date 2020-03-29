from .voc import VOCDataset
from .registry import DATASETS

@DATASETS.register_module
class defect(VOCDataset):
    
    CLASSES = ('bruise','dirty','scratch')
