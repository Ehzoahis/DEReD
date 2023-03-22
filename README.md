# DEReD
The Offical Codebase for Fully Self-Supervised Depth Estimation from Defocus Clue

## Run Training
```python
python scripts/train.py --data_path [path/to/dataset] --dataset [Dataset] --recon_all \ 
-N [experiment_name] --use_cuda -E 1000 --BS 32 --save_checkpoint --save_best --save_last \
--sm_loss_beta 2.5 --verbose --recon_loss_lambda 10e3 --aif_blur_loss_lambda 10 \
--blur_loss_lambda 1e1 --sm_loss_lambda 1e1 --log --vis
```

## Run Eval
```python
python scripts/train.py --data_path [path/to/dataset] --dataset [Dataset] --recon_all \
-N [experiment_name] --use_cuda --BS 32 --save_best --verbose --eval
```

## Model Weight
You can download the model weights trained on NYUv2 Focal Stack from [here](https://drive.google.com/file/d/1LQUt7Lo6KPKb0OsBETkvxeujOVdqOGjh/view?usp=share_link).

## Contact Authors
[Haozhe Si](mailto:haozhes3@illinois.edu), [Bin Zhao](mailto:zhaobin@pjlab.org.cn), [Dong Wang](mailto:wangdong@pjlab.org.cn), [Xuelong Li](mailto:li@nwpu.edu.cn)
