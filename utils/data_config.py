import sys
sys.path.append('..')

from .util_func import ThinLenCamera
from model import LearnableThinLens
import numpy as np

def get_data_config(args):
    if args.dataset == 'NYUv2':
        dataset_config = {
            'root_dir': args.data_path,
            'norm': args.normalize_dpt, 
            'shuffle': args.shuffle,
            'img_num': args.image_num, 
            'visible_img': args.visible_image_num,
            'focus_dist': [1, 3, 5, 7, 9],
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'scale': 1,
            'near': args.camera_near,
            'far': args.camera_far,
        }

    elif args.dataset == 'NYU100':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': args.shuffle,
            'img_num': 100, 
            'visible_img': args.visible_image_num,
            'focus_dist': np.linspace(1, 9, 100),
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'scale': 1,
            'near': args.camera_near,
            'far': args.camera_far,
        }

    elif args.dataset == 'mobileDFD':
        dataset_config = {
            'root_dir': args.data_path,
            'visible_img': args.visible_image_num,
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'scale': 1,
            'near': args.camera_near,
            'far': args.camera_far,
        }

    elif args.dataset == 'DefocusNet':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': args.shuffle,
            'img_num':  5, 
            'visible_img': 5,
            'focus_dist': [.3, .45, .75, 1.2, 1.8], 
            'recon_all': args.recon_all,
            'near': 0.1,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'far': 3.,
            'scale': args.scale
        }
    else:
        exit()
    return dataset_config

def get_camera(args):
    CameraModel = ThinLenCamera
    if args.dataset == 'NYUv2':
        camera = CameraModel(fnumber=0.5, focal_length=2.9*1e-3, pixel_size=5.6e-6)
    elif args.dataset == 'NYU100':
        camera = CameraModel(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
    elif args.dataset == 'mobileDFD':
        camera = CameraModel(fnumber=24, focal_length=50*1e-3, pixel_size=5.6e-6)
    elif args.dataset == 'DefocusNet':
        camera = CameraModel(fnumber=1.2, focal_length=7.2*1e-3, pixel_size=5.6e-6)
    else:
        camera = CameraModel(args.fnumber, args.focal_length, args.sensor_size, args.image_size)
    return camera
