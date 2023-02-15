import OpenEXR
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

def read_dpt(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.frombuffer(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt

root_dir = "/mnt/petrelfs/sihaozhe/my_fs_6/"
imglist_dpt = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-7:] == "Dpt.exr"]

for dpt in tqdm(imglist_dpt):
    prefix = dpt.split('.')[0]
    img_dpt_path = os.path.join(root_dir, dpt)
    depth = read_dpt(img_dpt_path)
    save_path = os.path.join(root_dir, prefix+'.npy')
    with open(save_path, 'wb') as f:
        np.save(f, depth[:, :, None])

imglist_aif = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-7:] == "Aif.tif"]
for img in tqdm(imglist_aif):
    prefix = img.split('.')[0]
    img_path = os.path.join(root_dir, img)
    im = Image.open(img_path)
    save_path = os.path.join(root_dir, prefix+'.png')
    im.save(save_path)

imglist_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-7:] == "All.tif"]
for img in tqdm(imglist_all):
    prefix = img.split('.')[0]
    img_path = os.path.join(root_dir, img)
    im = Image.open(img_path)
    save_path = os.path.join(root_dir, prefix+'.png')
    im.save(save_path)
