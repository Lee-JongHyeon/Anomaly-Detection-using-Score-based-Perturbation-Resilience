## Anomaly-Detection-using-Score-based-Perturbation-Resilience (ICCV 2023)

This repository contains the implementation for Anomaly Detection using Score-based Perturbation Resilience
[[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)

Unsupervised anomaly detection is widely studied in industrial applications where anomalous data is difficult to obtain. In particular, reconstruction-based anomaly detection can be a feasible solution if there is no option to use external knowledge, such as extra datasets or pre-trained models. However, reconstruction-based methods have limited utility due to poor detection performance. A scorebased model, also known as a denoising diffusion model, recently has shown a high sample quality in the generation task. In this paper, we propose a novel unsupervised anomaly detection method leveraging the score-based model. The proposed method shows promising performance without requiring external knowledge. The score, a gradient of the log-likelihood, has a property that is available for anomaly detection. The samples on the data manifold can be restored instantly by the score, even if they are randomly perturbed. We call this score-based perturbation resilience. On the other hand, the samples that deviate from the manifold cannot be restored in the same way. The variation of resilience depending on the sample position can be an indicator to discriminate anomalies. We derive this statement from a geometric perspective. Our method shows superior performance on three benchmark datasets for industrial anomaly detection. Specifically, on MVTec AD, we achieve image-level AUROC of 97.7% and pixel-level AUROC of 97.4% outperforming previous works that do not use external knowledge.

## Data Preparations
Download MVTEC dataset from [[Link]](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## Pretrained weights Preparations
Download pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1fvF1RFeOCWIraWhTUu71ZX1TX5Za8_kb?usp=drive_link)

