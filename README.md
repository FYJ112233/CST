# CST
# Cross-modality-Spatial-temporal-Transformer
This is the official implementation of our paper: Cross-modality Spatial-temporal Transformer for Video-based Visible-infrared Person Re-identification

### **1. Prepare the datasets.**

 HITSZ-VCM Dataset: The  HITSZ-VCM dataset can be downloaded from this [1] by submitting a copyright form.


### 2. Training.

  - `--dataset`: which dataset "VCM ".

  - `--lr`: initial learning rate.

  - `--gpu`:  which gpu to run.
  

First, you may need to manually define the data path. Then, you need run the code on 1 GPU of A6000 with 48G memory. 

### 3. Testing
The model in Baidu Netdisk: https://pan.baidu.com/s/1zYGCQWI9dvLOjZZcFA6nYw?, password: ggxv

### 4. Results.
|  Methods | Infrared - Visible | Visible - Infrared |
|----------|--------------------|--------------------|
|          | R1,  mAP            | R1,  mAP            |
| CAJL [3]     |  56.59,    41.49                  |   60.13,    42.81                 |
| MITML [1]    |   63.74,     45.31             |  64.54, 47.69                  |
| IBAN [2]     | 65.03,   48.77                 |   69.58,  50.96                |
| Our      |  69.44,  51.16                 |  72.64,     53.00              |


### 5. References

```
[1] Lin X, Li J, Ma Z, et al. Learning modal-invariant and temporal-memory for video-based visible-infrared person re-identification[C]. Computer Vision and Pattern Recognition. 2022: 20973-20982.
```

```
[2] Li H, Liu M, Hu Z, et al. Intermediary-guided Bidirectional Spatial-Temporal Aggregation Network for Video-based Visible-Infrared Person Re-Identification[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2023.
```

```
[3] Ye M, Ruan W, Du B, et al. Channel augmented joint learning for visible-infrared recognition[C]. International Conference on Computer Vision. 2021: 13567-13576.
```


### 6. Acknowledgments
The code was developed based on the CAJL [1] and MITML[3].  
Thanks for [1], [2], [3] providing visible-infrared reid code base and dataset.


