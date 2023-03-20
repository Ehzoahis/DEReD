# DEReD
The Offical Codebase for Fully Self-Supervised Depth Estimation from Defocus Clue

## Authors
[Haozhe Si](haozhes3@illinois.edu), [Bin Zhao](zhaobin@pjlab.org.cn), [Dong Wang](wangdong@pjlab.org.cn), [Yunpeng Gao](marennqx@gmail.com), [Mulin Chen](chenmulin@pjlab.org.cn), [Zhigang Wang](wangzhigang@pjlab.org.cn), [Xuelong Li](li@nwpu.edu.cn), 

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
