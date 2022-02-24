# Calculate-TRE-for-Medical-Image-Registration

we performed a method for calculate TRE for medical image registration.

本代码的大部分内容来自 https://github.com/vincentme/GroupRegNet

# cal_TRE

朱振宇基于上述文档，完成了适用于pairwise的TRE计算方法

纪宇整合相关内容并添加了部分注释

# get_dvf

## 本代码实现功能：

**已有推理完成的形变场，提取源图像中itk坐标靶点的形变距离**

## 流程如下：

1. 根据itk原点、图像分辨率信息，将ITK坐标转换为pixel坐标
2. 根据裁剪信息，将原始（512\*512\*66）坐标转换为裁剪（400\*260\*66）后的坐标
3. 将形变场从disp_f2m转为disp_m2f，即将STN使用的后向映射转为前向映射disp
4. 根据形变场构建一个采样网格，根据坐标采样形变场大小
5. 将像素形变场转为itk形变场

## 需要注意的问题

- 形变场 disp 的尺寸 [3, 66, 260, 400]和坐标信息是相反的，用下边方法解决

```
    pred = inter(np.flip(coordinate_crop_pixel,0))
```

- pred的3个维度和坐标信息的是一致的

