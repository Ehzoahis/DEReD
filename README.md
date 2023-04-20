# DEReD (Depth Estimation via Reconstucting Defocus Image)

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://ehzoahis.github.io/DeReD)
[![arXiv](https://img.shields.io/badge/arXiv-2303.11791-b31b1b.svg)](https://arxiv.org/pdf/2303.10752.pdf)

Official codes of CVPR 2023 [Paper](https://arxiv.org/pdf/2303.10752.pdf) | _Fully Self-Supervised Depth Estimation from Defocus Clue_

## Prepreation

### Environment

Create a new environment and install dependencies with `requirement.txt`:

```shell
conda create -n dered

conda activate dered

conda install --file requirements.txt

python gauss_psf/setup.py install
```

### Data 

The generation code for NYUv2 Focal Stack dataset is provided.

The generation code for DefocusNet can be found [here](https://github.com/dvl-tum/defocus-net).


### Weight
You can download the model weights trained on NYUv2 Focal Stack from [here](https://drive.google.com/file/d/1LQUt7Lo6KPKb0OsBETkvxeujOVdqOGjh/view?usp=share_link).


## Usage

### Train

```shell
python scripts/train.py --data_path [path/to/dataset] --dataset [Dataset] --recon_all \ 
-N [experiment_name] --use_cuda -E 1000 --BS 32 --save_checkpoint --save_best --save_last \
--sm_loss_beta 2.5 --verbose --recon_loss_lambda 10e3 --aif_blur_loss_lambda 10 \
--blur_loss_lambda 1e1 --sm_loss_lambda 1e1 --log --vis
```

### eval

```shell
python scripts/train.py --data_path [path/to/dataset] --dataset [Dataset] --recon_all \
-N [experiment_name] --use_cuda --BS 32 --save_best --verbose --eval
```

## Acknowledgement
Parts of the code are developed from [DefocusNet](https://github.com/dvl-tum/defocus-net) and [UnsupervisedDepthFromFocus](https://github.com/shirgur/UnsupervisedDepthFromFocus).

## Ciatation

```bibtex
@article{si2023fully,
  title={Fully Self-Supervised Depth Estimation from Defocus Clue},
  author={Si, Haozhe and Zhao, Bin and Wang, Dong and Gao, Yupeng and Chen, Mulin and Wang, Zhigang and Li, Xuelong},
  journal={arXiv preprint arXiv:2303.10752},
  year={2023}
}
```

## Contact Authors
[Haozhe Si](mailto:haozhes3@illinois.edu), [Bin Zhao](mailto:zhaobin@pjlab.org.cn), [Dong Wang](mailto:wangdong@pjlab.org.cn), [Xuelong Li](mailto:li@nwpu.edu.cn)