# YOLO11L 水稻病虫害检测模型卡（项目侧）

## 来源

- 发布方：Data-Juicer
- Hugging Face：`datajuicer/YOLO11L-Rice-Disease-Detection`
- 权重 SHA-256：`f1fb26e50cf571ac44d8ee9c0599d6331011d1f8bb1f2a98e4a626d9ccf282ad`

## 发布方报告信息

- 架构：YOLO11L
- 数据量：3,567 张水稻图像及标注
- 训练轮数：200 epochs
- 测试集 mAP50：56.3
- 测试集 mAP50-95：34.9

上述指标来自发布方模型卡，不是本项目独立复现实验结果。

## 类别

0. 水稻白叶枯病 `Bacterial_Leaf_Blight`
1. 水稻胡麻斑病 `Brown_Spot`
2. 健康水稻 `HealthyLeaf`
3. 稻瘟病 `Leaf_Blast`
4. 模型原标签：水稻叶鞘腐病 `Leaf_Scald`
5. 水稻窄褐斑病 `Narrow_Brown_Leaf_Spot`
6. 水稻穗颈瘟 `Neck_Blast`
7. 稻飞虱/稻铁甲类标签 `Rice_Hispa`

> 注意：模型卡第 4 类中文写作“叶鞘腐病”，英文为 `Leaf_Scald`。
> 二者在植病学中通常不是同一病害。本项目内部始终使用稳定代码 `leaf_scald`，
> 展示时注明“沿用模型原标签”，避免擅自更改模型语义。

## 限制

- 类别范围固定，类别之外的问题可能被误判成最相似的已有类别。
- 光照、背景、遮挡、清晰度和拍摄距离可能影响结果。
- 目标检测置信度不等同于农业诊断概率。
- 发布方数据分布可能与实际田间环境不同。
