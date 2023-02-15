import sys
sys.path.append('..')

from utils import *
from model import *
from tqdm import trange
import cv2

render = GaussPSF(7)
render.cuda()

camera = ThinLenCamera(8, focal_length=50e-3, pixel_size=1.2e-5)
data_path = '/mnt/lustre/sihaozhe/data/NYUv2-ori'
fd_list = [1, 1.5, 2.5, 4, 6]

for split in ['train', 'test']:
    rgb_path = os.path.join(data_path,f'{split}_rgb')
    dpt_path = os.path.join(data_path, f'{split}_depth')
    fs_path = os.path.join(data_path, f'{split}_fs{len(fd_list)}')
    
    if not os.path.exists(fs_path):
        os.mkdir(fs_path)

    imglist = [f for f in os.listdir(dpt_path) if os.path.isfile(os.path.join(dpt_path, f))]
    imglist.sort()

    for idx in trange(len(imglist)):
        exp_aif = cv2.resize(cv2.imread(os.path.join(rgb_path, imglist[idx])), (640//2, 480//2))/255.
        exp_dpt = Image.open(os.path.join(dpt_path, imglist[idx]))
        exp_dpt = np.asarray(exp_dpt, dtype=np.float32)
        exp_dpt = np.clip(np.asarray(exp_dpt, dtype=np.float32) / 1e4, 0.1, 10)
        exp_dpt = cv2.resize(exp_dpt, (640//2, 480//2))
        
        dpt = torch.from_numpy(exp_dpt).unsqueeze(0)
        aif = torch.from_numpy(exp_aif.transpose(2, 0, 1)).type(torch.float32).contiguous()
        
        for i, fd in enumerate(fd_list):
            defocus = camera.getCoC(dpt, fd).type(torch.float32)
            fs = render(aif.unsqueeze(0).cuda(), defocus.cuda())
            im = fs.cpu().numpy()[0].transpose(1, 2, 0) * 255.
            fn = f'{imglist[idx][:-4]}_{str(i).zfill(4)}.png'
            cv2.imwrite(os.path.join(fs_path, fn), im.astype(np.uint8))