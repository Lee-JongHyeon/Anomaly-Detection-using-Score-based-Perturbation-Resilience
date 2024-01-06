## Anomaly-Detection-using-Score-based-Perturbation-Resilience

This repository contains the implementation for Anomaly Detection using Score-based Perturbation Resilience
[[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)

## Environments 
- Python 3.8
- CUDA 11.3
- Packages:
```bash
Pillow==8.4.0
numpy==1.19.5
scikit-learn==0.24.2
torch-ema==0.3
torch==1.12.0
torchvision==0.13.0
```
## Data Preparations
Download MVTEC dataset from [[Link]](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## Train
```bash
python main.py --dataset_path ./mvtec/    \
               --save_path ./save/        \
               --class_name all

```
## Pretrained weights
Download pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1fvF1RFeOCWIraWhTUu71ZX1TX5Za8_kb?usp=drive_link)

## Citation

``` bibtex
@inproceedings{Anomaly-Detection-using-Score-based-Perturbation-Resilience,
  title={Anomaly Detection using Score-based Perturbation Resilience},
  author={Shin, Woosang and Lee, Jonghyeon and Lee, Taehan and Lee, Sangmoon and Yun, Jong Pil},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23372--23382},
  year={2023}
}
```
## Acknowledgement
Our repository is inspired by the following repositories. Thank you for their contribution.
- [Score SDE](https://github.com/yang-song/score_sde)
- [[MahalanobisAD]](https://github.com/byungjae89/MahalanobisAD-pytorch)
- [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet)

