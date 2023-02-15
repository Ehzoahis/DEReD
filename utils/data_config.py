import sys
sys.path.append('..')

from .util_func import ThinLenCamera
from model import LearnableThinLens
import numpy as np

def get_data_config(args):
    if args.dataset == 'NYUv2':
        if args.normalize_dpt:
            fd = [0.2, 0.3, 0.4, 0.65, 0.95]
        else:
            # fd = [1, 1.5, 2.5, 4, 6]
            fd = [1, 3, 5, 7, 9]

        dataset_config = {
            'root_dir': args.data_path,
            'norm': args.normalize_dpt, 
            'shuffle': args.shuffle,
            'img_num': args.image_num, 
            'visible_img': args.visible_image_num,
            'focus_dist': fd,
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

    elif args.dataset == 'DSLR':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': args.shuffle,
            'img_num':  5, 
            'visible_img': 5,
            'focus_dist': [1, 1.5, 2.5, 4, 6],
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'near': args.camera_near,
            'far':args.camera_far,
            'scale':1,
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

    elif args.dataset == 'SC':
        assert args.fd != None
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': False,
            'img_num':  5, 
            'visible_img': 5,
            'focus_dist': args.fd,
            'recon_all': args.recon_all,
            'near': args.camera_near,
            'far':args.camera_far,
            'scale':10,
        }

#[0.1, .15, .3, 0.7, 1.5], 
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
        if not args.normalize_dpt:
            # camera = CameraModel(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
            camera = CameraModel(fnumber=8.0, focal_length=50*1e-3, pixel_size=1.2e-5)
        else:
            camera = CameraModel(fnumber=0.5, focal_length=2.9*1e-3, pixel_size=5.6e-6)
    elif args.dataset == 'NYU100':
        camera = CameraModel(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
        # print('Modified Camera!!!!!')
        # camera = CameraModel(fnumber=8.0, focal_length=50*1e-3, pixel_size=1.2e-5)
    elif args.dataset == 'DSLR':
        camera = CameraModel(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
    elif args.dataset == 'SC':
        camera = CameraModel(fnumber=args.fnumber, focal_length=50*1e-3, pixel_size=6.5e-5)
    elif args.dataset == 'mobileDFD':
        camera = CameraModel(fnumber=24, focal_length=50*1e-3, pixel_size=5.6e-6)
    elif args.dataset == 'DefocusNet':
        # camera = CameraModel(fnumber=1.2, focal_length=2.9*1e-3, pixel_size=5.6e-6)
        # camera = CameraModel(fnumber=2.4, focal_length=2.9*1e-3, pixel_size=5.6e-6)
        camera = CameraModel(fnumber=1.2, focal_length=7.2*1e-3, pixel_size=5.6e-6)
    else:
        camera = CameraModel(args.fnumber, args.focal_length, args.sensor_size, args.image_size)
    return camera
