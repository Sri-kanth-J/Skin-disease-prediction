import shutil
import kagglehub
import os
os.environ["KAGGLEHUB_CACHE"]=r""
# Download latest version
path = kagglehub.dataset_download("riyaelizashaju/skin-disease-classification-image-dataset")
base="/mnt/d/Code/Skin-disease-prediction/datasets"
# base=r"D:\Code\Skin-disease-prediction\datasets"
datapath=base+r"/riyaelizashaju/skin-disease-classification-image-dataset/versions/1/Split_smol"
destination= base
for fold in os.listdir(datapath):
    src=os.path.join(datapath,fold)
    dst=os.path.join(destination,fold)
    if not os.path.exists(dst):
        shutil.move(src, dst)
shutil.rmtree(base+r"/riyaelizashaju")