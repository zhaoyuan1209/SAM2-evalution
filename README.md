# SAM2-evaluation

预处理数据集(修改文件夹目录root，和保存的目录save)
```
python preprocess/mvtec.py
```

一个例子生成预测
--mode支持auto/bbox/point三种模式，--model_type包含sam-l/sam2-l，--output预测图保存的路径，最终路径会在后面加上'_auto/bbox/point'因此每个数据集设置一个路径即可，根据模式自动改名。
--input 测试集图片，--gtpath 测试集真值

```
python scripts/prediction_img.py --mode point --model_type sam-l --output prediction/SAMl/SOD/DUTS --input datasets/DUTS/DUTS-TE/images --gtpath datasets/DUTS/DUTS-TE/gt
```

或者 一种更简单的例子生成预测

```
python scripts/run_generate_image.py
```

最后，通过scripts_cal_metric/predict_*_score*.py生成任务指标结果的预测（需要修改py文件内的预测路径predict_path和预处理后保存的数据集路径source_path），命令如下：
```
python scripts_cal_metric/predict_ad_score.py
```

项目涉及到的数据集如下：

| Datasets                                                                                                                                            |    Test Sets     |              Tasks               |                                                                 Remark                                                                  |
|:----------------------------------------------------------------------------------------------------------------------------------------------------|:----------------:|:--------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
| <br> [**MVTec-AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad) <br>                                                                  |   1725 images    |        Anomaly Detection         |                                                                                                                                         |
| [**VisA**](https://github.com/amazon-science/spot-diff)                                                                                             |   2162 images    |        Anomaly Detection         |                                                                                                                                         |
| [**MVTec-3D**](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)                                                                         |   1197 images    |        Anomaly Detection         |                                                           Only Use RGB images                                                           |
| [**BTAD**](https://ieeexplore.ieee.org/abstract/document/9576231)                                                                                   |    741 images    |        Anomaly Detection         |                                                                                                                                         |
| [**X-ray PBD**](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD)                                                                                      |    600 images    |        Automatic Power Battery Detection         |                                                                                                                                         |
| [**Brats2020**](https://www.med.upenn.edu/cbica/brats2020/)                                                                                         |  37 MRI images   | 3D Lesion Segmentation  | Test 4 modalities separately: Flair, T1ce, T1, T2， divided into training/validation/testing sets as per [**literature**](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_32#Sec5) — 315/16/37. |
| [**ISBI2015**](https://smart-stats-tools.org/lesion-challenge-2015)                                                                                 | 2162 MRI  images |    3D Lesion Segmentation        |                                                                                                                                         |



 